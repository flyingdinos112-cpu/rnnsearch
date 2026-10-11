"""Train RNNencdec on English→French (Bahdanau et al., ICLR 2015).

Paper setup (Appendix B): Adadelta (ε=1e-6, ρ=0.95), L2 gradient clip of 1,
minibatch of 80, sentences up to 50 tokens, 30k vocab, embed 620, hidden 1000.
Defaults below are a laptop-scale demo of the same architecture.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from dataset import EOS, PAD, SOS, prepare_data
from rnnenc50 import Decoder, Encoder, RNNenc50


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--maxout-dim", type=int, default=128)
    p.add_argument("--vocab-size", type=int, default=8000)
    p.add_argument("--max-len", type=int, default=20)
    p.add_argument("--teacher-forcing", type=float, default=0.5)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--save", type=Path, default=Path("checkpoints/rnnenc50.pt"))
    p.add_argument("--paper", action="store_true", help="Use paper hyperparameters (very large).")
    p.add_argument("--wmt14", action="store_true")
    p.add_argument("--wmt-max-pairs", type=int, default=50_000)
    p.add_argument("--val-ratio", type=float, default=0.0)
    return p.parse_args()


def apply_paper_hparams(args):
    args.embed_dim = 620
    args.hidden_dim = 1000
    args.maxout_dim = 500
    args.vocab_size = 30000
    args.max_len = 50
    args.batch_size = 80
    return args


def build_model(src_vocab_size, tgt_vocab_size, args, device):
    encoder = Encoder(src_vocab_size, args.embed_dim, args.hidden_dim, PAD)
    decoder = Decoder(tgt_vocab_size, args.embed_dim, args.hidden_dim, args.maxout_dim, PAD)
    return RNNenc50(encoder, decoder, device, SOS).to(device)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, n_tokens = 0.0, 0
    for src, src_len, tgt, _ in loader:
        src, src_len, tgt = src.to(device), src_len.to(device), tgt.to(device)
        logits = model(src, src_len, tgt, teacher_forcing_ratio=1.0)
        loss = criterion(logits[:, 1:].reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
        n = (tgt[:, 1:] != PAD).sum().item()
        total += loss.item() * n
        n_tokens += n
    return total / max(n_tokens, 1)


def main():
    args = parse_args()
    if args.paper:
        apply_paper_hparams(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_vocab, tgt_vocab, train_loader, val_loader = prepare_data(
        vocab_size=args.vocab_size,
        max_len=args.max_len,
        batch_size=args.batch_size,
        use_wmt14=args.wmt14,
        wmt_max_pairs=args.wmt_max_pairs,
        val_ratio=args.val_ratio,
    )
    model = build_model(len(src_vocab), len(tgt_vocab), args, device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    # Paper: Adadelta with ε=1e-6, ρ=0.95
    optimizer = torch.optim.Adadelta(model.parameters(), eps=1e-6, rho=0.95)

    print(
        f"device={device}  train_batches={len(train_loader)}  "
        f"src_vocab={len(src_vocab)}  tgt_vocab={len(tgt_vocab)}  "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )

    best_val = float("inf")
    args.save.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running, n_tokens = 0.0, 0
        for src, src_len, tgt, _ in train_loader:
            src, src_len, tgt = src.to(device), src_len.to(device), tgt.to(device)
            optimizer.zero_grad()
            logits = model(src, src_len, tgt, teacher_forcing_ratio=args.teacher_forcing)
            loss = criterion(logits[:, 1:].reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            n = (tgt[:, 1:] != PAD).sum().item()
            running += loss.item() * n
            n_tokens += n

        train_nll = running / max(n_tokens, 1)
        if len(val_loader) > 0:
            val_nll = evaluate(model, val_loader, criterion, device)
            score, label = val_nll, f"val_nll={val_nll:.3f}"
        else:
            score, label = train_nll, "no_val"
        print(
            f"epoch {epoch:02d}  train_nll={train_nll:.3f}  {label}  "
            f"time={time.time() - t0:.1f}s"
        )
        if score < best_val:
            best_val = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "src_itos": src_vocab.itos,
                    "tgt_itos": tgt_vocab.itos,
                    "args": vars(args),
                },
                args.save,
            )

    print(f"saved best checkpoint to {args.save} (score={best_val:.3f})")


if __name__ == "__main__":
    main()
