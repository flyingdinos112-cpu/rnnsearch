"""Re-score a saved RNNenc50 checkpoint on its validation split.

Rebuilds the exact train/val split used in training (same data args and
shuffle seed) and reports per-token NLL and perplexity, both teacher-forced
(comparable to train_nll) and free-running (decoder fed its own predictions).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn

from dataset import PAD, prepare_data
from translate import load_checkpoint


@torch.no_grad()
def score(model, loader, criterion, device, teacher_forcing_ratio):
    model.eval()
    total, n_tokens = 0.0, 0
    for src, src_len, tgt, _ in loader:
        src, src_len, tgt = src.to(device), src_len.to(device), tgt.to(device)
        logits = model(src, src_len, tgt, teacher_forcing_ratio=teacher_forcing_ratio)
        loss = criterion(logits[:, 1:].reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
        n = (tgt[:, 1:] != PAD).sum().item()
        total += loss.item() * n
        n_tokens += n
    return total / max(n_tokens, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/rnnenc50.pt"))
    parser.add_argument("--batch-size", type=int, default=None, help="Override the training batch size.")
    parser.add_argument("--skip-free-running", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, src_vocab, tgt_vocab, ns = load_checkpoint(args.ckpt, device)
    if not ns.val_ratio:
        raise SystemExit(f"{args.ckpt} was trained with --val-ratio 0; there is no validation split.")

    data_src_vocab, data_tgt_vocab, _, val_loader = prepare_data(
        vocab_size=ns.vocab_size,
        max_len=ns.max_len,
        batch_size=args.batch_size or ns.batch_size,
        use_wmt14=ns.wmt14,
        wmt_max_pairs=ns.wmt_max_pairs,
        opus100_max_pairs=getattr(ns, "opus100_max_pairs", 0),
        phrasebook_repeat=getattr(ns, "phrasebook_repeat", 0),
        val_ratio=ns.val_ratio,
    )
    if data_src_vocab.itos != src_vocab.itos or data_tgt_vocab.itos != tgt_vocab.itos:
        raise SystemExit("Rebuilt vocabulary differs from the checkpoint's; the data split did not reproduce.")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    print(f"loaded {args.ckpt} on {device}  val_batches={len(val_loader)}")

    tf_nll = score(model, val_loader, criterion, device, teacher_forcing_ratio=1.0)
    print(f"teacher-forced  val_nll={tf_nll:.3f}  ppl={math.exp(tf_nll):.1f}")
    if not args.skip_free_running:
        fr_nll = score(model, val_loader, criterion, device, teacher_forcing_ratio=0.0)
        print(f"free-running    val_nll={fr_nll:.3f}  ppl={math.exp(fr_nll):.1f}")


if __name__ == "__main__":
    main()
