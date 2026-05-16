# 🏏 IPL Match Predictor

> Predict the likely winner of an IPL match from first-innings statistics using a scikit-learn classifier deployed via Streamlit.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)](https://streamlit.io)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-brightgreen)](https://ipl-match-predictor-1.streamlit.app/)

**Live app → https://ipl-match-predictor-1.streamlit.app/**

---

## Overview

This project predicts the winner of an IPL match based on the batting team's first-innings performance. Given runs scored, wickets lost, overs bowled, and toss information, the model outputs a predicted winner and per-team win probabilities.

The model is trained on historical IPL match and delivery data (`matches.csv`, `deliveries.csv`) and saved as a scikit-learn pipeline (`ipl_model.pkl`). A Streamlit frontend (`app.py`) loads the pipeline and runs inference in real time.

---

## Repository Structure
```
ipl-match-predictor/
├── app.py                  # Streamlit UI and inference logic
├── main.ipynb              # Full training pipeline (EDA → features → model → eval)
├── ipl_model.pkl           # Trained scikit-learn pipeline (see Model section)
├── matches.csv             # Raw match-level data
├── deliveries.csv          # Raw ball-by-ball delivery data
├── requirements.txt        # Python dependencies
└── README.md
```
---
## How It Works

### Input
The user provides:
| Field | Type | Description |
|---|---|---|
| Batting Team | string | Team batting first |
| Bowling Team | string | Team bowling first |
| Runs Scored | int | First-innings runs at the point of prediction |
| Wickets Lost | int | Wickets fallen so far |
| Overs Bowled | float | Overs completed (used to compute run rate) |
| Won Toss? | bool | Whether the batting team won the toss |

### Feature Construction
`app.py` derives the following features before passing to the model:

| Feature | Derivation |
|---|---|
| `First_Inning_Run` | Raw input |
| `First_Inning_Wicket` | Raw input |
| `First_Inning_RunRate` | `runs / overs` |
| `Batting_Team` | Raw input (label-encoded inside pipeline) |
| `Bowling_Team` | Raw input (label-encoded inside pipeline) |
| `toss_decision` | `"bat"` or `"field"` derived from toss winner |
| `Toss_Winner_Batted` | `1` if batting team won toss, else `0` |

### Model
See `main.ipynb` for the full training walkthrough. Summary:

- **Algorithm:** Random Forest Classifier
- **Encoding:** `LabelEncoder` for `Batting_Team` and `Bowling_Team` (fitted on training data and saved inside the pipeline)
- **Train/test split:** 80/20, stratified by `winner`
- **Data range:** IPL seasons 2008–2022
- **Test accuracy:** 72.4%
- **AUC-ROC:** 0.79

Confusion matrix and feature importance plots are available at the bottom of `main.ipynb`.

### Output
- Predicted winner
- Win probability for each team (from `predict_proba`)

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/iankitsubedi/IPL-Match-Predictor.git
cd IPL-Match-Predictor
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

`ipl_model.pkl` is included in the repo, so no training step is required to run the app.

### 3. Retrain the model (optional)

Open and run `main.ipynb` end-to-end. The final cell saves a new `ipl_model.pkl` to the repo root.

```bash
jupyter notebook main.ipynb
```

---

## Model Card

| Property | Detail |
|---|---|
| Model type | Random Forest Classifier (`n_estimators=100`) |
| Training data | IPL 2008–2022 (`matches.csv` + `deliveries.csv`) |
| Target variable | Match winner (binary: batting team wins or not) |
| Test accuracy | 72.4% |
| AUC-ROC | 0.79 |
| Categorical encoding | `LabelEncoder` (fitted and saved in pipeline) |
| Known limitations | Does not account for player form, injuries, or pitch conditions. Accuracy drops for teams with limited historical data (e.g. newer franchises). Mid-innings predictions assume a stable run rate. |

---

## Known Limitations & Edge Cases

- **Unseen team names** will cause a `LabelEncoder` error. The app validates input against the list of teams seen during training.
- **0 overs bowled** will produce a division-by-zero for run rate; the app clamps overs to a minimum of 0.1.
- **New IPL franchises** (added post-2022) are not in the training data and will not be recognised.
- This is a first-innings prediction tool. It does not update predictions during the second innings.

---

## Requirements
```
streamlit
scikit-learn
pandas
numpy
```

Full pinned versions in `requirements.txt`.

---

## Contact

**Ankit Subedi** · ankitsubedi98765@gmail.com
