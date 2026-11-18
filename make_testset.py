"""Build held-out EN→FR test sets as TSV files (English<TAB>French).

- newstest2014: the WMT '14 test set the RNNsearch paper reports BLEU on.
- opus100_test: OPUS-100's test split, mixed domains (subtitles, everyday text).

Training only ever reads the *train* splits of these corpora (see dataset.py),
so neither file overlaps the training data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path(__file__).resolve().parent / "data" / "test"


def clean(text: str) -> str:
    return " ".join(text.replace("\t", " ").split())


def write_tsv(rows, path: Path, skip_identical: bool = False) -> int:
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            en, fr = clean(row["translation"]["en"]), clean(row["translation"]["fr"])
            if not en or not fr or (skip_identical and en == fr):
                continue
            handle.write(f"{en}\t{fr}\n")
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wmt = load_dataset("wmt/wmt14", "fr-en", split="test")
    n = write_tsv(wmt, args.out_dir / "newstest2014.tsv")
    print(f"newstest2014: {n} pairs")

    opus = load_dataset("Helsinki-NLP/opus-100", "en-fr", split="test")
    # Untranslated copies are common in OPUS, as in training.
    n = write_tsv(opus, args.out_dir / "opus100_test.tsv", skip_identical=True)
    print(f"opus100_test: {n} pairs")


if __name__ == "__main__":
    main()
