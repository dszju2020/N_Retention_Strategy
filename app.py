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
# 3. Ranges
# ===============================
RANGES = {
    "Day": (0, 120),
    "Temperature": (10.77, 82.2),
    "MC": (11.01, 81.22),
    "pH": (4.35, 9.46),
    "EC": (0.44, 7.66),
    "TN": (0.78, 3.48)
}

# ===============================
# 4. Streamlit Web 
# ===============================
st.title("RL-Based Optimal Microbial Prediction Platform")

st.header("Please enter physical and chemical indicators")

Day = st.number_input(
    f"Day (range {RANGES['Day'][0]} - {RANGES['Day'][1]})", 
    min_value=0.0, value=35.0
)
Temperature = st.number_input(
    f"Temperature (℃) (range {RANGES['Temperature'][0]} - {RANGES['Temperature'][1]})", 
    min_value=0.0, value=34.7
)
MC = st.number_input(
    f"MC (%) (range {RANGES['MC'][0]} - {RANGES['MC'][1]})", 
    min_value=0.0, value=29.86
)
pH = st.number_input(
    f"pH (range {RANGES['pH'][0]} - {RANGES['pH'][1]})", 
    min_value=0.0, value=8.54
)
EC = st.number_input(
    f"EC (ms/cm) (range {RANGES['EC'][0]} - {RANGES['EC'][1]})", 
    min_value=0.0, value=3.73
)
TN = st.number_input(
    f"TN (%) (range {RANGES['TN'][0]} - {RANGES['TN'][1]})", 
    min_value=0.0, value=2.10
)

input_state = np.array([Day, Temperature, MC, pH, EC, TN])

# ===============================
# 5. Input range check
# ===============================
if st.button("Predict"):
    errors = []
    if not (RANGES["Day"][0] < Day < RANGES["Day"][1]):
        errors.append(f"Day must be in range {RANGES['Day']}, got {Day}")
    if not (RANGES["Temperature"][0] < Temperature < RANGES["Temperature"][1]):
        errors.append(f"Temperature must be in range {RANGES['Temperature']}, got {Temperature}")
    if not (RANGES["MC"][0] < MC < RANGES["MC"][1]):
        errors.append(f"MC must be in range {RANGES['MC']}, got {MC}")
    if not (RANGES["pH"][0] < pH < RANGES["pH"][1]):
        errors.append(f"pH must be in range {RANGES['pH']}, got {pH}")
    if not (RANGES["EC"][0] < EC < RANGES["EC"][1]):
        errors.append(f"EC must be in range {RANGES['EC']}, got {EC}")
    if not (RANGES["TN"][0] < TN < RANGES["TN"][1]):
        errors.append(f"TN must be in range {RANGES['TN']}, got {TN}")

    if errors:
        st.error("Error: The input value is beyond the model's prediction range.")
        for e in errors:
            st.warning(e)
    else:
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
