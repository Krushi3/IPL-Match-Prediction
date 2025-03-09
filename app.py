import streamlit as st
import pickle
import pandas as pd

# Define Teams and Cities
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# Load Model Safely
try:
    with open(r'F:\ipl-win-probability-predictor-main\pipe.pkl', 'rb') as f:
        pipe = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found! Please check the file path.")
    st.stop()

# Streamlit App Title
st.title('🏏 IPL Win Probability Predictor')

# Sidebar for Team Selection
st.sidebar.header("Select Teams & Match Details")
batting_team = st.sidebar.selectbox('Batting Team', sorted(teams))
bowling_team = st.sidebar.selectbox('Bowling Team', sorted(teams))

# Ensure Different Teams
if batting_team == bowling_team:
    st.sidebar.error("Batting and Bowling teams must be different.")
    st.stop()

# Host City Selection
selected_city = st.sidebar.selectbox('Host City', sorted(cities))

# Target Runs
target = st.sidebar.number_input('Target Score', min_value=1, step=1)

# Match Progress Inputs
col1, col2, col3 = st.columns(3)
score = col1.number_input('Current Score', min_value=0, step=1)
overs = col2.number_input('Overs Completed', min_value=0.0, max_value=20.0, step=0.1)
wickets = col3.number_input('Wickets Fallen', min_value=0, max_value=10, step=1)

# Validate Inputs
if score >= target:
    st.error("Score cannot be greater than or equal to the target.")
    st.stop()
if overs == 0:
    crr = 0  # Avoid division by zero
else:
    crr = score / overs

balls_left = int(120 - (overs * 6))
runs_left = target - score
remaining_wickets = 10 - wickets

if balls_left == 0:
    rrr = 0
else:
    rrr = (runs_left * 6) / balls_left

# Prediction Button
if st.button('Predict Probability'):
    # Prepare Input Data
    input_df = pd.DataFrame({
        'batting_team': [batting_team], 'bowling_team': [bowling_team],
        'city': [selected_city], 'runs_left': [runs_left],
        'balls_left': [balls_left], 'wickets': [remaining_wickets],
        'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]
    })
    
    # Make Prediction
    result = pipe.predict_proba(input_df)
    win_prob = result[0][1] * 100
    loss_prob = result[0][0] * 100

    # Display Results
    st.markdown(f"### **{batting_team}: {win_prob:.1f}% Win Probability** 🎯")
    st.markdown(f"### **{bowling_team}: {loss_prob:.1f}% Win Probability** 🏏")
