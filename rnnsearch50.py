"""RNNsearch from Bahdanau, Cho & Bengio (ICLR 2015), FILE_3201.pdf.

Unlike RNNenc-50, the decoder does not see one fixed vector c. It looks back
at every source word at every step ("soft-search"), Appendix A.2:

  Encoder (A.2.1): bidirectional GRU sharing one embedding matrix E.
    Annotation h_j = [→h_j ; ←h_j]  (2n-dim).

  Alignment (A.1.2): e_ij = v_a^T tanh(W_a s_{i-1} + U_a h_j)
    α_ij = softmax_j(e_ij),  c_i = Σ_j α_ij h_j.  U_a h_j is precomputed.

  Decoder (A.2.2): GRU with
    s̃_i = tanh(W E y_{i-1} + U[r_i ∘ s_{i-1}] + C c_i)   (and z_i, r_i alike)
    s_0 = tanh(W_s ←h_1)

  Deep output (A.2.2): t̃_i = U_o s_{i-1} + V_o E y_{i-1} + C_o c_i,
    maxout over pairs -> t_i (l-dim), p(y_i) ∝ exp(y_i^T W_o t_i).

Sizes (A.2.3): n = 1000, m = 620, l = 500, n' = 1000.
Initialisation follows B.1 (see init_paper_weights).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BiEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, pad_idx: int):
        super().__init__()
        # One embedding E shared by both directions; separate GRU weights per direction.
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, source_seq: torch.Tensor, source_lengths: torch.Tensor):
        embedded = self.embedding(source_seq)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, source_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, h_n = self.gru(packed)
        annotations, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True, total_length=source_seq.size(1)
        )
        # annotations: (batch, src_len, 2n). h_n[1] is the backward RNN's state
        # after reading back to x_1, i.e. ←h_1, used for s_0.
        mask = torch.arange(source_seq.size(1), device=source_seq.device)[None, :] < source_lengths[:, None]
        return annotations, h_n[1], mask


class Attention(nn.Module):
    """a(s_{i-1}, h_j) = v_a^T tanh(W_a s_{i-1} + U_a h_j)  (Appendix A.1.2)."""

    def __init__(self, hidden_dim: int, align_dim: int):
        super().__init__()
        self.W_a = nn.Linear(hidden_dim, align_dim, bias=False)
        self.U_a = nn.Linear(2 * hidden_dim, align_dim)
        self.v_a = nn.Linear(align_dim, 1, bias=False)

    def precompute(self, annotations: torch.Tensor) -> torch.Tensor:
        # U_a h_j does not depend on i, so compute it once per sentence.
        return self.U_a(annotations)

    def forward(self, s_prev, annotations, keys, mask):
        energy = self.v_a(torch.tanh(self.W_a(s_prev).unsqueeze(1) + keys)).squeeze(2)
        energy = energy.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(energy, dim=1)  # (batch, src_len)
        context = torch.bmm(alpha.unsqueeze(1), annotations).squeeze(1)  # (batch, 2n)
        return context, alpha


class AttnDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, maxout_dim, align_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.attention = Attention(hidden_dim, align_dim)
        # GRUCell on [E y_{i-1} ; c_i] gives the W·Ey + C·c terms in s̃, z and r.
        self.gru = nn.GRUCell(embed_dim + 2 * hidden_dim, hidden_dim)
        self.init_state = nn.Linear(hidden_dim, hidden_dim)  # W_s
        # t̃ = U_o s_{i-1} + V_o E y_{i-1} + C_o c_i  (2l outputs)
        self.maxout_in = nn.Linear(hidden_dim + embed_dim + 2 * hidden_dim, maxout_dim * 2)
        self.fc_out = nn.Linear(maxout_dim, vocab_size)  # W_o

    def init_hidden(self, backward_first: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.init_state(backward_first))  # s_0 = tanh(W_s ←h_1)

    def forward(self, prev_token, s_prev, annotations, keys, mask):
        # prev_token: (batch, 1); s_prev: (batch, n)
        embedded = self.embedding(prev_token.squeeze(1))
        context, alpha = self.attention(s_prev, annotations, keys, mask)
        s_new = self.gru(torch.cat((embedded, context), dim=1), s_prev)
        # Deep output uses s_{i-1}, as written in Appendix A.2.2.
        t_tilde = self.maxout_in(torch.cat((s_prev, embedded, context), dim=1))
        t = t_tilde.view(t_tilde.size(0), -1, 2).max(dim=2).values
        return self.fc_out(t), s_new, alpha


class RNNsearch50(nn.Module):
    def __init__(self, encoder: BiEncoder, decoder: AttnDecoder, device: torch.device, sos_idx: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.sos_idx = sos_idx

    def _encode(self, source, source_lengths):
        annotations, backward_first, mask = self.encoder(source, source_lengths)
        keys = self.decoder.attention.precompute(annotations)
        return self.decoder.init_hidden(backward_first), annotations, keys, mask

    def forward(self, source, source_lengths, target, teacher_forcing_ratio: float = 0.5):
        batch_size, trg_len = target.shape
        vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, trg_len, vocab_size, device=self.device)

        s, annotations, keys, mask = self._encode(source, source_lengths)
        input_token = target[:, 0].unsqueeze(1)
        for t in range(1, trg_len):
            prediction, s, _ = self.decoder(input_token, s, annotations, keys, mask)
            outputs[:, t] = prediction
            if torch.rand(1).item() < teacher_forcing_ratio:
                input_token = target[:, t].unsqueeze(1)
            else:
                input_token = prediction.argmax(1, keepdim=True)
        return outputs

    @torch.no_grad()
    def translate(self, source, source_lengths, max_len: int, eos_idx: int) -> torch.Tensor:
        """Greedy decoding: argmax at each step until <eos> or max_len."""
        self.eval()
        s, annotations, keys, mask = self._encode(source, source_lengths)
        input_token = torch.full((source.size(0), 1), self.sos_idx, dtype=torch.long, device=self.device)
        tokens = []
        for _ in range(max_len):
            prediction, s, _ = self.decoder(input_token, s, annotations, keys, mask)
            input_token = prediction.argmax(1, keepdim=True)
            tokens.append(input_token)
            if (input_token.squeeze(1) == eos_idx).all():
                break
        return torch.cat(tokens, dim=1)

    # --- interface used by beam_search.py -------------------------------
    def beam_init(self, source, source_lengths):
        return self._encode(source, source_lengths)

    def beam_step(self, tokens, state):
        s, annotations, keys, mask = state
        logits, s, _ = self.decoder(tokens, s, annotations, keys, mask)
        return logits, (s, annotations, keys, mask)

    @staticmethod
    def beam_reorder(state, index):
        return tuple(x.index_select(0, index) for x in state)


def init_paper_weights(model: RNNsearch50) -> None:
    """Appendix B.1: orthogonal recurrent weights; W_a, U_a ~ N(0, 0.001²);
    v_a and all biases zero; every other weight ~ N(0, 0.01²)."""
    attention = model.decoder.attention
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "bias" in name:
                nn.init.zeros_(p)
            elif "weight_hh" in name:  # U, U_z, U_r stacked as 3 (n x n) blocks
                for block in p.chunk(3, dim=0):
                    nn.init.orthogonal_(block)
            elif p is attention.W_a.weight or p is attention.U_a.weight:
                nn.init.normal_(p, 0.0, 0.001)
            elif p is attention.v_a.weight:
                nn.init.zeros_(p)
            else:
                nn.init.normal_(p, 0.0, 0.01)
        for emb in (model.encoder.embedding, model.decoder.embedding):
            emb.weight[emb.padding_idx].zero_()
