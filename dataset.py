"""Parallel EN–FR data and vocabularies for RNNencdec.

The paper trains on WMT '14 English–French (30k most-frequent words per
language, Moses tokenization, no lowercasing). Full WMT is hundreds of
millions of words; this module uses a small bundled phrase set so the
translator can be trained on a laptop. Pass --wmt14 to use Hugging Face
`wmt14` / `fr-en` if you want the paper's corpus.
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

SPECIALS = ("<pad>", "<sos>", "<eos>", "<unk>")
PAD, SOS, EOS, UNK = range(4)

DATA_FILE = Path(__file__).resolve().parent / "data" / "en_fr_pairs.tsv"
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Whitespace + punctuation split, close to Moses-style tokenization."""
    return TOKEN_RE.findall(text.strip())


def load_tsv_pairs(path: Path) -> List[Tuple[List[str], List[str]]]:
    pairs = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            src, tgt = line.split("\t", 1)
            src_toks = tokenize(src.lower())
            tgt_toks = tokenize(tgt.lower())
            if src_toks and tgt_toks:
                pairs.append((src_toks, tgt_toks))
    return pairs


def load_wmt14(max_pairs: int, max_len: int) -> List[Tuple[List[str], List[str]]]:
    from datasets import load_dataset

    raw = load_dataset("wmt/wmt14", "fr-en", split="train", streaming=True)
    pairs = []
    for row in raw:
        src_toks = tokenize(row["translation"]["en"])
        tgt_toks = tokenize(row["translation"]["fr"])
        if 1 <= len(src_toks) <= max_len and 1 <= len(tgt_toks) <= max_len:
            pairs.append((src_toks, tgt_toks))
        if len(pairs) >= max_pairs:
            break
    return pairs


class Vocab:
    def __init__(self, tokens: Iterable[str], max_size: int):
        counts = Counter(tokens)
        most_common = [tok for tok, _ in counts.most_common(max(0, max_size - len(SPECIALS)))]
        self.itos = list(SPECIALS) + most_common
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: Sequence[str], add_sos_eos: bool = True) -> List[int]:
        ids = [self.stoi.get(tok, UNK) for tok in tokens]
        if add_sos_eos:
            return [SOS] + ids + [EOS]
        return ids

    def decode(self, ids: Sequence[int], skip_specials: bool = True) -> str:
        words = []
        for idx in ids:
            if skip_specials and idx in (PAD, SOS, EOS):
                if idx == EOS:
                    break
                continue
            words.append(self.itos[idx] if 0 <= idx < len(self.itos) else "<unk>")
        return " ".join(words)


class ParallelDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Tuple[List[str], List[str]]],
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        max_len: int,
    ):
        self.examples = []
        for src, tgt in pairs:
            if len(src) > max_len or len(tgt) > max_len:
                continue
            self.examples.append((src_vocab.encode(src), tgt_vocab.encode(tgt)))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        src, tgt = self.examples[index]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def collate_batch(batch):
    src_list, tgt_list = zip(*batch)
    src_lengths = torch.tensor([len(s) for s in src_list], dtype=torch.long)
    tgt_lengths = torch.tensor([len(t) for t in tgt_list], dtype=torch.long)
    src = pad_sequence(src_list, batch_first=True, padding_value=PAD)
    tgt = pad_sequence(tgt_list, batch_first=True, padding_value=PAD)
    return src, src_lengths, tgt, tgt_lengths


def build_vocabs(pairs: Sequence[Tuple[List[str], List[str]]], vocab_size: int) -> Tuple[Vocab, Vocab]:
    src_tokens = [tok for src, _ in pairs for tok in src]
    tgt_tokens = [tok for _, tgt in pairs for tok in tgt]
    return Vocab(src_tokens, vocab_size), Vocab(tgt_tokens, vocab_size)


def prepare_data(
    vocab_size: int,
    max_len: int,
    batch_size: int,
    val_ratio: float = 0.1,
    seed: int = 42,
    use_wmt14: bool = False,
    wmt_max_pairs: int = 50_000,
):
    if use_wmt14:
        pairs = load_wmt14(wmt_max_pairs, max_len)
    else:
        pairs = load_tsv_pairs(DATA_FILE)
        pairs = [(s, t) for s, t in pairs if len(s) <= max_len and len(t) <= max_len]

    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val = int(len(pairs) * val_ratio)
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    if not train_pairs:
        train_pairs, val_pairs = pairs, []
    src_vocab, tgt_vocab = build_vocabs(train_pairs, vocab_size)

    train_ds = ParallelDataset(train_pairs, src_vocab, tgt_vocab, max_len)
    val_ds = ParallelDataset(val_pairs, src_vocab, tgt_vocab, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    return src_vocab, tgt_vocab, train_loader, val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wmt14", action="store_true")
    args = parser.parse_args()
    src_vocab, tgt_vocab, train_loader, _ = prepare_data(
        vocab_size=30000, max_len=50, batch_size=8, use_wmt14=args.wmt14
    )
    src, src_len, tgt, _ = next(iter(train_loader))
    print(f"src vocab {len(src_vocab)} | tgt vocab {len(tgt_vocab)}")
    print("sample src ids:", src[0].tolist())
    print("sample src:", src_vocab.decode(src[0].tolist(), skip_specials=False))
    print("sample tgt:", tgt_vocab.decode(tgt[0].tolist(), skip_specials=False))
    print("src length:", src_len[0].item())


if __name__ == "__main__":
    main()
