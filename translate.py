"""English→French translation with a trained RNNenc50 or RNNsearch50 checkpoint.

Uses beam search by default, as in the paper; --beam-size 1 gives greedy decoding.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from beam_search import beam_search
from dataset import EOS, Vocab, tokenize
from train import build_model


def vocab_from_itos(itos):
    vocab = Vocab([], max_size=4)
    vocab.itos = list(itos)
    vocab.stoi = {tok: i for i, tok in enumerate(vocab.itos)}
    return vocab


def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ns = argparse.Namespace(**ckpt["args"])
    src_vocab = vocab_from_itos(ckpt["src_itos"])
    tgt_vocab = vocab_from_itos(ckpt["tgt_itos"])
    model = build_model(len(src_vocab), len(tgt_vocab), ns, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, src_vocab, tgt_vocab, ns


def encode_source(sentence: str, src_vocab: Vocab, device: torch.device, lowercase: bool = True):
    # The bundled TSV corpus is lowercased; WMT14 keeps original casing.
    if lowercase:
        sentence = sentence.lower()
    ids = src_vocab.encode(tokenize(sentence))
    tensor = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    lengths = torch.tensor([len(ids)], dtype=torch.long, device=device)
    return tensor, lengths


def detokenize(text: str) -> str:
    return (
        text.replace(" ' ", "'")
        .replace(" ' ", "'")
        .replace(" - ", "-")
    )


def translate_sentence(
    model, src_vocab, tgt_vocab, sentence: str, max_len: int, device,
    lowercase: bool = True, beam_size: int = 10,
):
    src, lengths = encode_source(sentence, src_vocab, device, lowercase)
    if beam_size > 1:
        pred_ids = beam_search(model, src, lengths, beam_size, max_len, EOS)
    else:
        pred_ids = model.translate(src, lengths, max_len=max_len, eos_idx=EOS)[0].tolist()
    return detokenize(tgt_vocab.decode(pred_ids))


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/rnnenc50.pt"))
    parser.add_argument("--beam-size", type=int, default=10,
                        help="Beam width (the paper does not state one); 1 = greedy.")
    parser.add_argument("sentences", nargs="*", help="English sentences to translate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, src_vocab, tgt_vocab, ns = load_checkpoint(args.ckpt, device)

    sentences = args.sentences or [
        "hello",
        "i love you",
        "where is the hotel",
        "the weather is nice",
        "thank you",
        "i am a student",
    ]
    print(f"loaded {args.ckpt} ({getattr(ns, 'model', 'rnnenc')}) on {device}  beam={args.beam_size}")
    for sentence in sentences:
        french = translate_sentence(
            model, src_vocab, tgt_vocab, sentence, ns.max_len, device,
            lowercase=not getattr(ns, "wmt14", False), beam_size=args.beam_size,
        )
        print(f"{sentence}  →  {french}")


if __name__ == "__main__":
    main()
