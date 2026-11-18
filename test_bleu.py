"""Translate a held-out TSV test set (English<TAB>French) and report BLEU.

Two scores are printed:
- tokenized BLEU: hypotheses and references both split with dataset.tokenize,
  scored as-is (like the paper's multi-bleu.perl numbers);
- sacreBLEU: the standard 13a-tokenized score on detokenized text.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import sacrebleu
import torch

from beam_search import beam_search
from dataset import EOS, UNK, tokenize
from translate import detokenize, encode_source, load_checkpoint


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/rnnsearch50.pt"))
    parser.add_argument("--test", type=Path, default=Path("data/test/opus100_test.tsv"))
    parser.add_argument("--beam-size", type=int, default=10, help="1 = greedy.")
    parser.add_argument("--limit", type=int, default=0, help="Only score the first N pairs.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Write en / reference / hypothesis TSV here (default: next to the test file).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, src_vocab, tgt_vocab, ns = load_checkpoint(args.ckpt, device)
    lowercase = not getattr(ns, "wmt14", False)

    pairs = []
    with args.test.open(encoding="utf-8") as handle:
        for line in handle:
            if "\t" in line:
                en, fr = line.rstrip("\n").split("\t", 1)
                pairs.append((en, fr))
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"loaded {args.ckpt} ({ns.model}) on {device}  beam={args.beam_size}  test={args.test} ({len(pairs)} pairs)")

    hyps_tok, refs_tok, hyps_text, refs_text = [], [], [], []
    src_unk = src_tokens = 0
    t0 = time.time()
    for i, (en, fr) in enumerate(pairs, 1):
        src, lengths = encode_source(en, src_vocab, device, lowercase)
        ids = src[0, 1:-1].tolist()
        src_unk += sum(1 for x in ids if x == UNK)
        src_tokens += len(ids)
        max_len = max(ns.max_len, 2 * len(ids))
        if args.beam_size > 1:
            pred = beam_search(model, src, lengths, args.beam_size, max_len, EOS)
        else:
            pred = model.translate(src, lengths, max_len=max_len, eos_idx=EOS)[0].tolist()
        hyp = tgt_vocab.decode(pred)
        ref = " ".join(tokenize(fr.lower() if lowercase else fr))
        hyps_tok.append(hyp)
        refs_tok.append(ref)
        hyps_text.append(detokenize(hyp))
        refs_text.append(fr)
        if i % 200 == 0:
            print(f"  {i}/{len(pairs)}  {time.time() - t0:.0f}s")

    tok_bleu = sacrebleu.corpus_bleu(hyps_tok, [refs_tok], tokenize="none", force=True)
    std_bleu = sacrebleu.corpus_bleu(hyps_text, [refs_text], lowercase=lowercase)
    print(f"\ntokenized BLEU: {tok_bleu.score:.2f}  ({tok_bleu})")
    print(f"sacreBLEU 13a:  {std_bleu.score:.2f}")
    print(f"source <unk> rate: {src_unk / max(src_tokens, 1):.1%}   time: {time.time() - t0:.0f}s")

    out = args.out or args.test.with_name(f"{args.test.stem}.{args.ckpt.stem}.hyp.tsv")
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("english\treference\tmodel\n")
        for (en, fr), hyp in zip(pairs, hyps_text):
            handle.write(f"{en}\t{fr}\t{hyp}\n")
    print(f"translations written to {out}")


if __name__ == "__main__":
    main()
