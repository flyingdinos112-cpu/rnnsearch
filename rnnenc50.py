"""RNNencdec-50 from Bahdanau, Cho & Bengio (ICLR 2015).

FILE_3201.pdf (arXiv:1409.0473) describes two models. This file implements the
baseline RNN Encoder–Decoder (RNNencdec / RNNenc-50), not RNNsearch:

  Encoder: unidirectional GRU reads x_1..x_T into hidden states; the context
  vector c is the last hidden state h_T (Appendix A.2.2: "the model becomes
  RNN Encoder–Decoder if we fix c_i to h_Tx").

  Decoder: GRU with the *same* c injected at every step. Deep output with a
  maxout layer, then softmax over the target vocabulary.

Equations follow Appendix A.1.1 / A.2. PyTorch GRU is equivalent to Cho's
gated unit (update-gate polarity is inverted in naming only). Concatenating
the context onto the decoder input realises the paper's W e(y) + C c terms
inside the gates and candidate state.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, pad_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, source_seq: torch.Tensor, source_lengths: torch.Tensor) -> torch.Tensor:
        # source_seq: (batch, src_len) with src_len <= 50 in the paper
        embedded = self.embedding(source_seq)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, source_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        # hidden: (1, batch, hidden_dim) — the fixed-length context vector c
        return hidden


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        maxout_dim: int,
        pad_idx: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        # GRU sees [y_{t-1} embedding ; context c] at every step
        self.gru = nn.GRU(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        # Decoder start state s_0 = tanh(W_s c), as in Cho et al. / Appendix A.2.2
        self.init_state = nn.Linear(hidden_dim, hidden_dim)
        # Deep output: t̃ = U_o s + V_o e(y) + C_o c, then maxout, then softmax
        self.maxout_in = nn.Linear(hidden_dim + embed_dim + hidden_dim, maxout_dim * 2)
        self.fc_out = nn.Linear(maxout_dim, vocab_size)

    def init_hidden(self, context: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.init_state(context.squeeze(0))).unsqueeze(0)

    def forward(self, target_token: torch.Tensor, hidden: torch.Tensor, context: torch.Tensor):
        # target_token: (batch, 1)
        embedded = self.embedding(target_token)
        context_seq = context.permute(1, 0, 2)
        gru_input = torch.cat((embedded, context_seq), dim=2)
        output, hidden = self.gru(gru_input, hidden)

        combined = torch.cat((output.squeeze(1), embedded.squeeze(1), context_seq.squeeze(1)), dim=1)
        # Maxout: t_i[j] = max(t̃_{2j-1}, t̃_{2j})  (Appendix A.2.2)
        t_tilde = self.maxout_in(combined)
        t_maxout = t_tilde.view(t_tilde.size(0), -1, 2).max(dim=2).values
        prediction = self.fc_out(t_maxout)
        return prediction, hidden


class RNNenc50(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device, sos_idx: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.sos_idx = sos_idx

    def forward(
        self,
        source: torch.Tensor,
        source_lengths: torch.Tensor,
        target: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        batch_size, trg_len = target.shape
        vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, trg_len, vocab_size, device=self.device)

        context = self.encoder(source, source_lengths)
        hidden = self.decoder.init_hidden(context)
        input_token = target[:, 0].unsqueeze(1)

        for t in range(1, trg_len):
            prediction, hidden = self.decoder(input_token, hidden, context)
            outputs[:, t] = prediction
            if self.training and torch.rand(1).item() < teacher_forcing_ratio:
                input_token = target[:, t].unsqueeze(1)
            else:
                input_token = prediction.argmax(1, keepdim=True)

        return outputs

    @torch.no_grad()
    def translate(
        self,
        source: torch.Tensor,
        source_lengths: torch.Tensor,
        max_len: int,
        eos_idx: int,
    ) -> torch.Tensor:
        """Greedy decoding: argmax at each step until <eos> or max_len."""
        self.eval()
        context = self.encoder(source, source_lengths)
        hidden = self.decoder.init_hidden(context)
        batch_size = source.size(0)
        input_token = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long, device=self.device)
        tokens = []
        for _ in range(max_len):
            prediction, hidden = self.decoder(input_token, hidden, context)
            input_token = prediction.argmax(1, keepdim=True)
            tokens.append(input_token)
            if (input_token.squeeze(1) == eos_idx).all():
                break
        return torch.cat(tokens, dim=1)
