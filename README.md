IPL Match Predictor
===================

Overview
--------
Predict the likely winner of an IPL match from first-innings statistics. The project provides a Streamlit UI (`app.py`) that loads a pre-trained scikit-learn classifier (`ipl_model.pkl`) and returns a predicted winner and associated probability.

How it works
------------
- Input: user provides batting team, bowling team, runs scored, wickets lost, and whether the batting team won the toss.
- Feature construction: the app computes a run rate and assembles a single-row pandas DataFrame with the required columns.
- Model: a saved scikit-learn estimator is loaded from `ipl_model.pkl`. The estimator must implement `predict` and `predict_proba`.
- Output: the app shows the predicted winner and a confidence score, plus individual team win probabilities.

Repository contents
-------------------
- `app.py` : Streamlit frontend and inference flow.
- `matches.csv`, `deliveries.csv` : raw match and delivery-level data (used for training, optional at runtime).
- `ipl_model.pkl` : trained model (required at runtime; not included here).
- `requirements.txt` : Python dependencies.

Model interface and expected input
---------------------------------
The loaded model must accept a pandas DataFrame with these columns (single row) and support `predict` and `predict_proba`:

- `First_Inning_Run`
- `First_Inning_Wicket`
- `First_Inning_RunRate`
- `Batting_Team`
- `Bowling_Team`
- `toss_decision` (string: `bat` or `field`)
- `Toss_Winner_Batted` (0 or 1)

Setup and run
-------------
Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the app locally (ensure `ipl_model.pkl` is present in the repo root):

```bash
streamlit run app.py
```

Training (optional)
-------------------
Training scripts are not included. Use `matches.csv` and `deliveries.csv` to engineer features and train a scikit-learn classifier. Save the fitted estimator with `pickle.dump(model, open('ipl_model.pkl', 'wb'))`.

Deployment
----------
Deployed site: ADD_DEPLOYED_URL_HERE

Contact
-------
Name: Ankit Subedi 
Email: ankitsubedi98765@gmail.com

Notes and caveats
-----------------
- `ipl_model.pkl` is required to run predictions. Without it the app will error when loading the model.
- The frontend contains UI styling and copy in `app.py` that may include emoji characters; the README omits emoji by design.
