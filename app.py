import streamlit as st
import pandas as pd
import pickle

# Page config
st.set_page_config(
    page_title="IPL Match Predictor",
    page_icon="🏏",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    with open("ipl_model.pkl", "rb") as f:
        return pickle.load(f)

clf = load_model()

# Styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@400;500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'DM Sans', sans-serif;
        }
        .title {
            font-family: 'Bebas Neue', sans-serif;
            font-size: 3.5rem;
            letter-spacing: 3px;
            color: #F5A623;
            margin-bottom: 0;
        }
        .subtitle {
            color: #888;
            font-size: 0.95rem;
            margin-top: -8px;
            margin-bottom: 2rem;
        }
        .result-box {
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            border: 2px solid #F5A623;
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            margin-top: 1.5rem;
        }
        .result-label {
            font-family: 'Bebas Neue', sans-serif;
            font-size: 1.2rem;
            letter-spacing: 2px;
            color: #888;
        }
        .result-team {
            font-family: 'Bebas Neue', sans-serif;
            font-size: 2.8rem;
            letter-spacing: 2px;
            color: #F5A623;
        }
        .result-prob {
            font-size: 1rem;
            color: #ccc;
            margin-top: 0.3rem;
        }
        .stButton > button {
            background-color: #F5A623;
            color: #000;
            font-family: 'Bebas Neue', sans-serif;
            font-size: 1.2rem;
            letter-spacing: 2px;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 2rem;
            width: 100%;
            cursor: pointer;
        }
        .stButton > button:hover {
            background-color: #e09510;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">🏏 IPL PREDICTOR</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict who wins based on first innings performance</div>', unsafe_allow_html=True)

# Teams list
teams = sorted([
    'Chennai Super Kings', 'Deccan Chargers', 'Delhi Capitals',
    'Gujarat Lions', 'Gujarat Titans', 'Kochi Tuskers Kerala',
    'Kolkata Knight Riders', 'Lucknow Super Giants', 'Mumbai Indians',
    'Pune Warriors', 'Punjab Kings', 'Rajasthan Royals',
    'Rising Pune Supergiants', 'Royal Challengers Bengaluru',
    'Sunrisers Hyderabad'
])

# Inputs
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox("🏏 Batting Team (1st Innings)", teams)
with col2:
    bowling_team = st.selectbox("🎯 Bowling Team (1st Innings)", [t for t in teams if t != batting_team])

col3, col4 = st.columns(2)
with col3:
    runs = st.number_input("Runs Scored", min_value=0, max_value=300, value=160)
with col4:
    wickets = st.number_input("Wickets Lost", min_value=0, max_value=10, value=5)

# Single clean toss question
toss_won = st.selectbox("🪙 Did the Batting Team Win the Toss?", ["Yes", "No"])

# Predict
if st.button("PREDICT WINNER"):
    run_rate = runs / 20
    toss_decision = "bat" if toss_won == "Yes" else "field"
    toss_batted = 1 if toss_won == "Yes" else 0

    input_df = pd.DataFrame({
        "First_Inning_Run": [runs],
        "First_Inning_Wicket": [wickets],
        "First_Inning_RunRate": [run_rate],
        "Batting_Team": [batting_team],
        "Bowling_Team": [bowling_team],
        "toss_decision": [toss_decision],
        "Toss_Winner_Batted": [toss_batted]
    })

    prediction = clf.predict(input_df)[0]
    proba = clf.predict_proba(input_df)[0]

    winner = batting_team if prediction == 1 else bowling_team
    confidence = proba[prediction] * 100

    st.markdown(f"""
        <div class="result-box">
            <div class="result-label">PREDICTED WINNER</div>
            <div class="result-team">{winner}</div>
            <div class="result-prob">Confidence: {confidence:.1f}%</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Win Probability**")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(batting_team, f"{proba[1]*100:.1f}%")
    with col_b:
        st.metric(bowling_team, f"{proba[0]*100:.1f}%")
