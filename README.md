# 🏏 IPL Match Predictor

> Predict the likely winner of an IPL match from first-innings statistics using a scikit-learn pipeline deployed via Streamlit.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)](https://streamlit.io)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-brightgreen)](https://ipl-match-predictor-1.streamlit.app/)

**Live app → https://ipl-match-predictor-1.streamlit.app/**

---

## Overview

This project predicts the winner of an IPL match based on the batting team's first-innings performance. Given runs scored, wickets lost, and toss information, the model outputs a predicted winner and per-team win probabilities.

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
| Field | Type | Description |
|---|---|---|
| Batting Team | string | Team batting first |
| Bowling Team | string | Team bowling first |
| Runs Scored | int | Total first-innings runs |
| Wickets Lost | int | Wickets fallen in first innings |
| Won Toss? | bool | Whether the batting team won the toss |

### Feature Construction
`app.py` derives the following features before passing to the model:

| Feature | Derivation |
|---|---|
| `First_Inning_Run` | Raw input |
| `First_Inning_Wicket` | Raw input |
| `First_Inning_RunRate` | `runs / 20` (assumes full 20 overs) |
| `Batting_Team` | Raw input (one-hot encoded inside pipeline) |
| `Bowling_Team` | Raw input (one-hot encoded inside pipeline) |
| `toss_decision` | `"bat"` or `"field"` derived from toss input |
| `Toss_Winner_Batted` | `1` if batting team won toss, else `0` |

### Pipeline
The saved `ipl_model.pkl` is a full scikit-learn `Pipeline` with two steps:
- **Preprocessor** (`ColumnTransformer`): `StandardScaler` on numeric features, `OneHotEncoder` on categorical features
- **Classifier**: `RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_split=5, random_state=42)`

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

`ipl_model.pkl` is included in the repo — no training step required.

### 3. Retrain the model (optional)

Open and run `main.ipynb` end-to-end. The final cell saves a new `ipl_model.pkl` to the repo root.

```bash
jupyter notebook main.ipynb
```

---

## Model Card

| Property | Detail |
|---|---|
| Model type | `RandomForestClassifier` (`n_estimators=300, max_depth=6`) |
| Training data | IPL 2008–2022 (`matches.csv` + `deliveries.csv`) |
| Total matches | 1,095 |
| Train/test split | 80/20, `random_state=42` |
| Target variable | `1` if batting team (first innings) wins, else `0` |
| Encoding | `OneHotEncoder(handle_unknown="ignore")` for teams and toss |
| Scaling | `StandardScaler` on numeric features |
| **Test accuracy** | **67.6%** |
| **Train accuracy** | **74.1%** |
| **AUC-ROC** | **0.73** |
| **5-fold CV mean** | **66.4%** |
| Known limitations | Does not account for player form, injuries, pitch conditions, or actual overs bowled. Run rate is fixed at `runs / 20`. Accuracy drops for franchises with limited data (e.g. Kochi Tuskers, Pune Warriors). |

## Requirements
---
```
streamlit
scikit-learn
pandas
numpy
```
---
Full pinned versions in `requirements.txt`.

---
## Contact

**Ankit Subedi** · ankitsubedi98765@gmail.com
