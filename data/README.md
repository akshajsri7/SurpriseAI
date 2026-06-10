# Data

The model is trained on quarterly earnings data for **S&P 100** companies, pulled from the **Bloomberg Terminal** in DePaul's John L. Keeley Jr. Finance Lab via the Bloomberg Excel Add-in. Each ticker produces **two CSV files**:

- `{TICKER}_ERN.csv` — the earnings history (one row per quarter)
- `{TICKER}_DETAIL.csv` — the per-analyst standing EPS estimates (one row per analyst per quarter)

The two are merged on `(ticker, period)` in [`notebooks/surpriseAI.ipynb`](../notebooks/surpriseAI.ipynb).

> ⚠️ **The data is NOT redistributable.** The Bloomberg license does not permit redistribution, so no CSVs are committed to this repository. `data/*.csv` is git-ignored. The schemas below let you reproduce the dataset from your own Bloomberg subscription.

---

## `{TICKER}_ERN.csv` — earnings history

Exported via the **ERN** function. One row per fiscal quarter. Column headers are read case-insensitively (the notebook lowercases and strips them). A trailing **"Average of Absolute Values"** summary row is exported by Bloomberg and is dropped during cleaning.

| Bloomberg column | Internal name | Type | Description |
|---|---|---|---|
| `Ann Date` | `ann_date` | date | Earnings announcement date. Rows with a blank `Ann Date` or blank `Reported` are dropped. |
| `Per` | `period` | string | Fiscal period label, e.g. `Q1 25` or `2025 Q1`. Normalized to a single canonical form so the two files join. |
| `Reported` | `reported` | float | Actual reported EPS for the quarter. |
| `Estimate` | `consensus` | float | Analyst consensus EPS (the benchmark expectation). |
| `%Surp` | `pct_surp` | percent | Earnings surprise as a percent of consensus. Parsed from a percent string. **Prediction target.** Clipped to ±50% during processing. |
| `%Px Chg` | `pct_px_chg` | percent | Post-announcement price change (%). Secondary target; clipped to ±30%. |
| `T12M` | `t12m` | float | Trailing-twelve-month figure. Optional; filled with `NaN` if absent. |
| `P/E` | `pe` | float | Price-to-earnings ratio. Optional; filled with `NaN` if absent. |

---

## `{TICKER}_DETAIL.csv` — per-analyst estimates

Exported via the live `=BQL(...)` broker-detail query (the estimates-consensus detail / **EEB** screen). One row per analyst per quarter — the **standing** estimate snapshot, *not* a time-ordered revision history. Coverage averages **~4 analysts per quarter**, mostly mid-tier and boutique firms.

| Bloomberg column | Internal name | Type | Description |
|---|---|---|---|
| `#EST().PERIOD` | `period` | string | Fiscal period label; normalized to match the ERN file and used as the join key. |
| `#EST().FIRM_NAME` | `firm_name` | string | Name of the analyst's firm (e.g. *D.A. Davidson*). |
| `#EST().ANALYST_NAME` | `analyst_name` | string | Name of the individual analyst. |
| `#EST().VALUE` | `estimate_value` | float | That analyst's EPS estimate for the period. |
| `#EST().EST_SOURCE_SPECIFIED` | `firm_code` | string | Bloomberg broker contributor code (e.g. `DAD` = D.A. Davidson, `GSR` = Goldman Sachs Research). An empty result for a broker code means that broker does not cover the stock — treated as missing data, not zero. |

The analyst embedding is keyed on the analyst/firm identity from this file. A single shared index is reserved for analysts unseen in the training set.

---

## Resulting dataset

After cleaning and the inner merge on `(ticker, period)`:

| | Count |
|---|---|
| Stocks | 92 |
| Quarter-announcements | 1,766 |
| Analyst-estimate rows | 6,936 (~4 / quarter) |
| Time-ordered split (train / val / test) | 1,219 / 261 / 267 |

The dataset is heavily skewed — **87.6% of surprises are positive** — which is why directional accuracy is reported alongside balanced accuracy in the results. See the project [README](../README.md) for the full discussion.

## Reproducing the data layout

Place your own exports in this directory, one pair per ticker:

```
data/
├── AAPL_ERN.csv
├── AAPL_DETAIL.csv
├── MSFT_ERN.csv
├── MSFT_DETAIL.csv
└── ...
```

The notebook discovers tickers automatically by globbing `*_ERN.csv`.
