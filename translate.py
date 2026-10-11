"""Greedy English→French decoding with a trained RNNenc50 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

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


def encode_source(sentence: str, src_vocab: Vocab, device: torch.device):
    ids = src_vocab.encode(tokenize(sentence.lower()))
    tensor = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    lengths = torch.tensor([len(ids)], dtype=torch.long, device=device)
    return tensor, lengths


def detokenize(text: str) -> str:
    return (
        text.replace(" ' ", "'")
        .replace(" ' ", "'")
        .replace(" - ", "-")
    )


def translate_sentence(model, src_vocab, tgt_vocab, sentence: str, max_len: int, device):
    src, lengths = encode_source(sentence, src_vocab, device)
    pred_ids = model.translate(src, lengths, max_len=max_len, eos_idx=EOS)
    return detokenize(tgt_vocab.decode(pred_ids[0].tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/rnnenc50.pt"))
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
    print(f"loaded {args.ckpt} on {device}")
    for sentence in sentences:
        french = translate_sentence(model, src_vocab, tgt_vocab, sentence, ns.max_len, device)
        print(f"{sentence}  →  {french}")


if __name__ == "__main__":
    main()
