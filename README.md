
# SurpriseAI

Can a neural network beat Wall Street's consensus? We train a transformer on analyst revision sequences to predict whether companies will beat or miss earnings expectations.

---

## Overview

When a company reports earnings, prices move based on the *gap* between actual results and analyst expectations - not the raw numbers. The standard benchmark, the analyst consensus, is a simple average that throws away information about who made each forecast and when.

**SurpriseAI** processes the full history of analyst forecast revisions before each earnings event using a transformer model with learned analyst embeddings. We test whether this approach predicts earnings surprises better than the consensus average.

---

## Team

| Name | Email |
|------|-------|
| Rayyan Hussain | rhussai9@depaul.edu |
| Akshaj Sriachutananda | asriachu@depaul.edu |
| Damian Moreno | dmoren37@depaul.edu |

---

## Project Structure

```
SurpriseAI/
├── data/
│   └── README.md           # Bloomberg field descriptions (data not included)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_transformer_model.ipynb
│   └── 04_ablation_analysis.ipynb
├── src/
│   ├── dataset.py          # Data loading & sequence construction
│   ├── model.py            # Transformer architecture
│   ├── baselines.py        # Zero, consensus, linear regression
│   └── metrics.py          # MAE, directional accuracy, DM test
├── results/
│   └── figures/            # Plots and tables
├── requirements.txt
└── README.md
```

---

## Methodology

**Input**: Each analyst revision in the 90 days before an earnings announcement is one element in the input sequence, combining a learned analyst embedding with features like estimate value, revision size, and days until earnings.

**Model**: A transformer encoder with two output heads — one predicting the earnings surprise, one predicting the post-announcement stock return.

**Baselines**:
- Predict zero surprise
- Latest consensus value
- Linear regression on revision summary statistics

**Key analyses**:
- Embedding ablation (does analyst identity matter?)
- Attention analysis (which revisions does the model rely on?)

---

## Data

Five years of quarterly earnings events for S&P 100 companies, pulled from the Bloomberg Terminal at DePaul University. Approximately 100,000 revision records across 2,000 earnings events.

> Data cannot be redistributed due to Bloomberg license restrictions.

---

## Results

*Coming soon — updated after final evaluation.*

| Model | MAE (¢/share) | Directional Accuracy |
|-------|--------------|----------------------|
| Zero baseline | — | — |
| Latest consensus | — | — |
| Linear regression | — | — |
| **SurpriseAI (ours)** | — | — |

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/SurpriseAI.git
cd SurpriseAI
pip install -r requirements.txt
```

Run notebooks in order inside `notebooks/`.

---

## Timeline

| Milestone | Deadline |
|-----------|----------|
| Data pull & cleaning | May 7, 2026 |
| Baselines | May 14, 2026 |
| Transformer model | May 21, 2026 |
| Ablation & attention analysis | May 28, 2026 |
| Final report | June 4, 2026 |

---

*DePaul University — Deep Learning Project, Spring 2026*
