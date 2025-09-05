# app.py
import streamlit as st
import numpy as np
import joblib

# ===============================
# 1. Loading models and standardizers
# ===============================
sclf_stage1 = joblib.load("models/sclf_stage1.pkl")
sclf_stage2 = joblib.load("models/sclf_stage2.pkl")
scaler = joblib.load("models/scaler.pkl")

ACTION_SPACE = np.array([0, 10, 20, 30, 40, 50, 60, 70])  # micro 

# ===============================
# 2. Define environment function
# ===============================
def environment(state, action_micro):
    state_scaled = scaler.transform(state.reshape(1, -1))
    X_stage2 = np.hstack([state_scaled, np.array([action_micro]).reshape(1, -1)])
    reward = sclf_stage2.predict(X_stage2)[0]
    return reward

def find_best_micro(state):
    best_micro = None
    best_y2 = -np.inf
    for micro in ACTION_SPACE:
        reward = environment(state, micro)
        if reward > best_y2:
            best_y2 = reward
            best_micro = micro
    return best_micro, best_y2

# ===============================
# 3. Streamlit Web 
# ===============================
st.title("RL-Based Optimal Microbial Prediction Platform")

st.header("Please enter physical and chemical indicators")

Day = st.number_input("Day", min_value=0.0, value=35.0)
Temperature = st.number_input("Temperature (℃)", min_value=0.0, value=34.7)
MC = st.number_input("MC (%)", min_value=0.0, value=29.86)
pH = st.number_input("pH", min_value=0.0, value=8.54)
EC = st.number_input("EC (ms/cm)", min_value=0.0, value=3.73)
TN = st.number_input("TN (%)", min_value=0.0, value=2.10)

input_state = np.array([Day, Temperature, MC, pH, EC, TN])

if st.button("Predict"):
    # ---------- Stage 1 - Predicted microbial abundance ----------
    state_scaled = scaler.transform(input_state.reshape(1, -1))
    micro_pred = sclf_stage1.predict(state_scaled)[0]

    # ---------- Stage 2 - Predicted Lable ----------
    X_stage2 = np.hstack([state_scaled, np.array([micro_pred]).reshape(1, -1)])
    y2_pred = sclf_stage2.predict(X_stage2)[0]

    # ---------- RL ----------
    best_micro, best_y2 = find_best_micro(input_state)

    # ---------- Output result ----------
    st.subheader("Prediction Results")
    st.success(f"Predicted microbial abundance (%): {micro_pred:.4f}")
    st.success(f"Predicted lable (score): {y2_pred:.4f}")
    st.success(f"RL Optimal microbial abundance (%): {best_micro}")
    st.success(f"RL Optimal lable (score): {best_y2:.4f}")
