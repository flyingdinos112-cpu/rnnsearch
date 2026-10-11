"""Build an RNNencdec-50 model with paper or demo sizes.

Paper (Bahdanau et al. ICLR 2015, RNNencdec-50):
  vocab 30_000, embed 620, hidden 1000, maxout 500, max length 50.
"""

from __future__ import annotations

import argparse

import torch

from dataset import PAD, SOS
from rnnenc50 import Decoder, Encoder, RNNenc50


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--paper", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.paper:
        vocab_size, embed_dim, hidden_dim, maxout_dim = 30000, 620, 1000, 500
    else:
        vocab_size, embed_dim, hidden_dim, maxout_dim = 8000, 128, 256, 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = Encoder(vocab_size, embed_dim, hidden_dim, PAD)
    dec = Decoder(vocab_size, embed_dim, hidden_dim, maxout_dim, PAD)
    model = RNNenc50(enc, dec, device, SOS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"RNNenc-50 initialized on {device}  params={n_params:,}  paper={args.paper}")


if __name__ == "__main__":
    main()
