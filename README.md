# SurpriseAI

**Can a transformer that learns *who* each analyst is predict earnings surprises better than the Wall Street consensus?**

A deep-learning study testing whether per-analyst embeddings can extract signal that the consensus average throws away. The honest answer, at the data depth we could obtain, is **no** — and this repo documents that negative result carefully, with three baselines, an LSTM comparison, a direct embedding ablation, an attention analysis, and proper significance testing.

> DePaul University · CSC 383 Deep Learning · Spring 2026

## Team

| Name | Email |
|---|---|
| Rayyan Hussain | rhussai9@depaul.edu |
| Akshaj Sriachutananda | asriachu@depaul.edu |
| Damian Moreno | dmoren37@depaul.edu |

The three of us shared the work across data extraction, modeling, and experiments/writing, with weekly check-ins.

---

## Concepts in plain terms

This project sits at the intersection of finance and machine learning, so here is the vocabulary you need — no background in either assumed.

**The finance side**

- **EPS (earnings per share)** — a company's quarterly profit divided by its number of shares: the profit attributable to one share.
- **Analyst consensus** — before a company reports, professional analysts at banks and research firms each publish their own guess for that EPS. The *consensus* is just the average of those guesses — the market's standard expectation.
- **Earnings surprise** — the gap between what the company *actually* reports and what the consensus expected, written as a percentage. A positive surprise means the company beat expectations; negative means it missed. **This is what we try to predict**, because a stock moves on the surprise, not on the raw number.
- **Bulge-bracket vs. mid-tier analysts** — "bulge-bracket" are the largest, most prestigious banks (Goldman Sachs, Morgan Stanley, JPMorgan); their analysts' reputations are strongest. "Mid-tier" and boutique firms are smaller. *Which* analysts cover a stock turns out to matter for our story.

**The machine-learning side**

- **Transformer** — the neural-network family behind tools like ChatGPT. Its key trick is **attention**: it can look at a whole group of inputs at once and decide, for each one, how much weight to give every other input. Here the inputs are the analyst guesses for one quarter, and attention lets the model decide which analysts to lean on.
- **Embedding** — a learned list of numbers the model attaches to each analyst (the way a language model attaches a vector to each word). The idea is that the embedding captures an analyst's track record and tendencies, so the model can learn that some analysts are sharper or more biased than others. **Whether these embeddings actually help is the core question of the project.**
- **Baseline** — a deliberately simple reference model. If a fancy model can't beat the simple ones, the fancy model isn't earning its keep.

**The core idea in one sentence:** the consensus average throws away *who* made each estimate and *how* their guesses are spread out — we test whether a transformer that keeps that detail, and learns each analyst's identity, can predict surprises better than the average and better than simpler models.

---

## The question

When a company reports quarterly earnings, the stock moves on the *gap* between the reported number and what analysts expected — the **earnings surprise** — not on the raw figure. The standard expectation is the **analyst consensus**, a plain average of individual broker estimates. Averaging discards information: it treats a stale forecast like a fresh one, a sloppy analyst like a sharp one, and smooths over bursts of revisions.

Prior work shows the consensus is biased in predictable ways and that analyst identity carries information. So we asked a narrow, concrete question:

> If we give a model the **individual** analyst estimates instead of their average, and let it **learn who each analyst is** via a learned embedding, can it beat the consensus and simpler models at predicting earnings surprises?

This is a test of a hypothesis, not a bid to build a winning trading model.

---

## The result (honest, and negative)

The transformer **cannot predict the size of a surprise any better than guessing the mean** (R² ≈ 0). It produces a **faint ranking signal** (information coefficient ≈ 0.11) that beats the linear baseline ~5×, but the signal is marginal and from a single run. Crucially, **turning the analyst-identity embeddings on or off makes no significant difference** — a direct negative answer to the central question. A transformer, an LSTM, and a ridge regression all land in roughly the same place.

The limiting factor is **data depth, not the model**: we could extract only ~4 mid-tier analysts per quarter as a *snapshot*, not the 30–80 time-ordered revisions (including bulge-bracket firms) the proposal called for. At that depth the individual estimates and the consensus are nearly the same object, so it is unsurprising that identity buys little.

### Test-set results

Test set = **267 announcements**, strictly time-ordered (train on the earliest events, test on the latest). RMSE and MAE are in percentage points of surprise. IC is the Pearson correlation between predicted and actual surprise; it is undefined for constant-forecast baselines.

| Model | RMSE | MAE | DirAcc | BalAcc | IC |
|---|---|---|---|---|---|
| **Transformer (ours)** | **12.44** | **7.54** | 0.876 | 0.50 | **0.111** |
| LSTM | 12.44 | 7.72 | 0.876 | 0.50 | 0.044 |
| Ridge (summary stats) | 13.60 | 8.55 | 0.873 | 0.498 | 0.019 |
| Predict-mean | 12.43 | 7.74 | 0.876 | 0.50 | — |
| Predict-zero / consensus | 14.10 | 9.16 | 0.00 | 0.00 | — |

**How to read the table**

- **Magnitude is a tie.** The transformer's RMSE (12.44) matches predict-mean (12.43); R² = −0.002. A Diebold–Mariano test against predict-mean gives **p = 0.80** — a clean tie. It beats predict-zero, but that is a low bar (predict-zero ignores the obvious positive skew).
- **Ranking is the only signal.** IC = 0.111 (p = 0.07, marginal) vs. ridge's 0.019 (indistinguishable from noise) — about **5× better ranking** than the linear model. Faint and from a single seed, so we treat it as suggestive, not established.
- **Directional accuracy 0.876 is a mirage.** 87.6% of surprises are positive, so a model that always says "beat" scores 0.876. **Balanced accuracy = 0.50** (a coin flip) exposes this. We report both side by side on purpose.
- **Identity adds nothing.** With vs. without analyst embeddings, test errors are essentially identical (MSE ≈ 154 both ways); a paired Wilcoxon test on per-event squared errors finds no significant difference.

---

## Why the negative result

Three model families (transformer, LSTM, ridge) converging to the same poor performance points at the **data**, not the architecture:

1. **Shallow analyst coverage.** ~4 standing estimates per quarter instead of the proposed 30–80-revision history. With so few estimates, the per-estimate detail and the consensus average carry nearly the same information.
2. **No bulge-bracket names.** Coverage is dominated by mid-tier and boutique firms. The analysts whose identity would most plausibly carry signal — Goldman, Morgan Stanley, JPMorgan — were not extractable through our subscription tier, so the embeddings had little genuine identity information to learn.
3. **Snapshots, not sequences.** What we could pull was the standing estimate per quarter, with **no ordering**. This is why the framing is **cross-sectional** rather than a revision-sequence model, and why **positional encoding was removed** — position carries no meaning in an unordered snapshot.
4. **Diffuse attention.** The summary token spreads weight across analysts (top analyst ≈ 0.15), with no single name dominating — consistent with the weak embedding result.

The negative embedding result should be read as **"identity does not help at this data depth,"** not as proof that it never could. The most promising next step is about data, not architecture: fuller revision-level history with bulge-bracket coverage and per-revision timestamps would give the same model a fair test of the original hypothesis.

---

## Methodology

**Input representation.** Each earnings event is one example: the set of analyst estimates for that quarter. Each estimate becomes a vector concatenating a learned **16-d analyst embedding** with four continuous features — the estimate value, its distance from the quarter mean, that distance normalized by the spread of estimates, and the prior quarter's reported figure. Continuous features are standardized using **training-set statistics only** (no test leakage).

**Architecture.**
1. For each analyst, concatenate continuous features with that analyst's embedding (20-d combined input).
2. Project into a **64-d** model space and prepend a learnable summary (CLS) token.
3. Pass through a **2-layer transformer encoder, 4 attention heads**, 128-d feedforward, with a padding mask so empty slots are ignored. **No positional encoding** (unordered snapshot).
4. Read the summary token into two heads: one predicts the percent surprise, one predicts the post-earnings return (down-weighted to half as a light secondary signal).

**Training.** Huber loss (threshold 5), **AdamW** (lr 1e-4, weight decay 1e-2), **one-cycle** schedule over **50 epochs**, gradient-norm clipping at 1.0, early stopping. **76,690 parameters.** Unseen analysts map to a single reserved index. Trained in PyTorch on a Colab A100 (a full run takes minutes).

**Baselines.**
- **Predict-zero** — assume no surprise (the consensus at face value).
- **Predict-mean** — guess the average training surprise for every event.
- **Ridge** — linear regression with an L2 penalty on per-quarter summary statistics; the key baseline, since it asks whether a transformer is needed at all.
- **LSTM** — a 2-layer bidirectional recurrent net on the same inputs, isolating whether *attention specifically* helps.

**Evaluation.** Strictly time-ordered split. We report RMSE, MAE, R², the information coefficient (Pearson and Spearman), directional **and** balanced accuracy, a **Diebold–Mariano** test (transformer vs. each baseline), an **embedding ablation** compared with a paired **Wilcoxon** signed-rank test, and an **attention analysis** of the summary-token weights.

---

## Data

Source: quarterly earnings data for **S&P 100** companies, pulled from the **Bloomberg Terminal** in DePaul's John L. Keeley Jr. Finance Lab via the Excel Add-in. Each ticker yields two CSVs — an earnings-history file (`{TICKER}_ERN.csv`) and a per-analyst estimate file (`{TICKER}_DETAIL.csv`).

After cleaning: **92 stocks**, **1,766 quarter-announcements**, **6,936 analyst-estimate rows** (~4 analysts/quarter, ~19 usable quarters/stock). The data is heavily skewed — **87.6% of surprises are positive**. Surprises are clipped to ±50% so a few extreme quarters (e.g. COVID) cannot dominate the loss. Time-ordered split: **1,219 / 261 / 267** (train / val / test).

> **The data is not redistributable** under the Bloomberg license. See [`data/README.md`](data/README.md) for the full field schemas; the CSVs themselves are not included and are git-ignored.

---

## Repository structure

```
SurpriseAI/
├── README.md                      # this file
├── requirements.txt               # torch, numpy, pandas, scikit-learn, scipy, matplotlib
├── .gitignore                     # guards against committing licensed data/*.csv
├── notebooks/
│   └── surpriseAI.ipynb           # end-to-end: data prep, baselines, model, ablation, plots
├── paper/
│   └── SurpriseAI_FinalPaper.pdf  # final write-up
└── data/
    └── README.md                  # Bloomberg CSV schemas (data not included)
```

---

## Running it

```bash
git clone https://github.com/akshajsri7/SurpriseAI.git
cd SurpriseAI
pip install -r requirements.txt
jupyter notebook notebooks/surpriseAI.ipynb
```

The notebook is the full pipeline — data cleaning and feature engineering, the baselines, the transformer, the embedding ablation, the significance tests, and the diagnostic plots (predicted-vs-actual scatter, error distribution, surprise distribution). Because the Bloomberg data cannot be redistributed, place your own `{TICKER}_ERN.csv` / `{TICKER}_DETAIL.csv` files under `data/` to reproduce the results; the schemas they must follow are documented in [`data/README.md`](data/README.md).

---

## References

1. Lawrence D. Brown and Emad Mohammad. *The predictive value of analyst characteristics.* Journal of Accounting, Auditing and Finance, 18(4), 2003.
2. Michael B. Clement and Senyo Y. Tse. *Financial analyst characteristics and herding behavior in forecasting.* The Journal of Finance, 60(1):307–341, 2005.
3. Amanda Cowen, Boris Groysberg, and Paul Healy. *Which types of analyst firms are more optimistic?* Journal of Accounting and Economics, 41(1-2):119–146, 2006.
4. Sepp Hochreiter and Jürgen Schmidhuber. *Long short-term memory.* Neural Computation, 9(8):1735–1780, 1997.
5. Scott E. Stickel. *Reputation and performance among security analysts.* The Journal of Finance, 47(5):1811–1836, 1992.
6. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. *Attention is all you need.* In Advances in Neural Information Processing Systems, volume 30, 2017.
