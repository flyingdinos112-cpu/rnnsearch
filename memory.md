# Memory: RNNencdec English–French translator

Date: 2026-10-11

## Paper

Read `FILE_3201.pdf`: Bahdanau, Cho, and Bengio, *Neural Machine Translation by Jointly Learning to Align and Translate* (ICLR 2015, arXiv:1409.0473).

Two models in the paper:

- **RNNencdec / RNNenc-50** — baseline RNN encoder–decoder (Cho et al.). The encoder GRU compresses the whole English sentence into one fixed vector \(c = h_T\). The decoder GRU generates French, injecting the **same** \(c\) at every step. That fixed-length bottleneck is why BLEU falls on long sentences.
- **RNNsearch** — bidirectional encoder plus attention. Not implemented here.

Paper RNNenc-50 settings (Appendix A–B): 30k vocab, embed 620, hidden 1000, maxout 500, max length 50, Adadelta (\(\epsilon=10^{-6}\), \(\rho=0.95\)), L2 grad clip 1, batch 80, WMT ’14 EN–FR.

## What was built

A working **English→French RNNencdec** (not attention), matching those equations.

| File | Role |
|---|---|
| `rnnenc50.py` | GRU encoder, GRU decoder with context at every step, maxout deep output, greedy `translate()` |
| `dataset.py` | Moses-like tokenize, vocabs, pad/collate, TSV loader; optional `--wmt14` |
| `data/en_fr_pairs.tsv` | ~250 phrase pairs for laptop training |
| `train.py` | Adadelta + clip; `--paper` for paper sizes; `--wmt14` for Hugging Face WMT |
| `translate.py` | Load checkpoint, greedy decode |
| `init.py` | Instantiate a model (`--paper` for full size) |
| `requirements.txt` | `torch`, `datasets` |
| `checkpoints/rnnenc50.pt` | Trained demo weights |

Existing stubs (`rnnenc50.py`, `init.py`, `dataset.py`) were completed. PyTorch was installed in `.venv`.

Demo hyperparameters (CPU): embed 128, hidden 256, maxout 128, max length 20, batch 16. Trained 80 epochs on the full phrasebook (`--val-ratio 0`, teacher forcing 0.7). Train NLL ~0.002.

## How to run

```bash
source .venv/bin/activate
python train.py --epochs 80 --teacher-forcing 0.7
python translate.py "how are you"
python train.py --paper --wmt14   # paper scale; needs GPU
```

## Sample translations (phrasebook model)

- hello → bonjour
- i love you → je t'aime
- where is the hotel → où est l'hôtel
- the weather is nice → il fait beau
- i would like a coffee → je voudrais un café

This checkpoint memorizes the bundled phrases. Open-domain WMT quality needs `--paper --wmt14` and days of GPU time, as in the paper.
