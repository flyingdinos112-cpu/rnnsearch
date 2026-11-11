"""Beam search decoding (Bahdanau et al. 2015, Sec. 4.2).

"Once a model is trained, we use a beam search to find a translation that
approximately maximizes the conditional probability" — following Sutskever
et al. (2014): keep the `beam_size` most probable partial translations; when
one emits <eos> it is set aside as finished and the beam shrinks by one.
The highest-probability finished translation wins. The paper does not give a
beam size or any length normalisation, so scores are plain sums of log-probs.

Works with any model that provides beam_init / beam_step / beam_reorder
(RNNenc50 and RNNsearch50 both do).
"""

from __future__ import annotations

from typing import List

import torch


@torch.no_grad()
def beam_search(model, source, source_lengths, beam_size: int, max_len: int, eos_idx: int) -> List[int]:
    """Translate one sentence (source has batch size 1). Returns target ids incl. final <eos>."""
    model.eval()
    device = source.device
    state = model.beam_init(source, source_lengths)

    hyps: List[List[int]] = [[]]
    scores = torch.zeros(1, device=device)
    tokens = torch.full((1, 1), model.sos_idx, dtype=torch.long, device=device)
    finished: List[tuple] = []  # (score, ids)

    for _ in range(max_len):
        logits, state = model.beam_step(tokens, state)
        log_probs = torch.log_softmax(logits, dim=-1)  # (live, vocab)
        vocab = log_probs.size(1)
        k = beam_size - len(finished)
        flat = (scores[:, None] + log_probs).view(-1)
        top_scores, top_idx = flat.topk(min(k, flat.numel()))

        keep_rows, keep_tokens, new_hyps, new_scores = [], [], [], []
        for score, idx in zip(top_scores.tolist(), top_idx.tolist()):
            row, tok = divmod(idx, vocab)
            ids = hyps[row] + [tok]
            if tok == eos_idx:
                finished.append((score, ids))
            else:
                keep_rows.append(row)
                keep_tokens.append(tok)
                new_hyps.append(ids)
                new_scores.append(score)

        if len(finished) >= beam_size or not new_hyps:
            break
        index = torch.tensor(keep_rows, dtype=torch.long, device=device)
        state = model.beam_reorder(state, index)
        tokens = torch.tensor(keep_tokens, dtype=torch.long, device=device).unsqueeze(1)
        hyps, scores = new_hyps, torch.tensor(new_scores, device=device)

    if not finished:  # hit max_len: fall back to the best unfinished hypothesis
        finished = [(s, h) for s, h in zip(scores.tolist(), hyps)]
    return max(finished, key=lambda x: x[0])[1]
