import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="IAQ Exceedance Predictor", layout="centered")

st.title("🔬 IAQ Exceedance Predictor")
st.markdown(
    "This tool predicts whether indoor air quality parameters will exceed safe thresholds "
    "based on selected occupant behavior and environmental conditions."
)

# --- Input Form ---
st.header("📥 Input Factors")

with st.form("iaq_input_form"):
    col1, col2 = st.columns(2)

    with col1:
        occupancy_label = st.selectbox("Number of People in the Room", ["One resident", "More than one residents"])
        occupancy = 1 if occupancy_label == "One resident" else 2

        activity = st.selectbox("Activity Type", [
            "Sleeping", "Sitting", "Watching TV", "Working",
            "Having Dinner", "Cleaning", "Dressing", "Cooking", "Smoking"
        ])
        window = st.selectbox("Window Status", ["Closed", "Open"])
        ac = st.selectbox("Air Conditioning (A/C)", ["Off", "On"])
        ach = st.selectbox("Air Change Rate (ACH)", ["Meet Thai regulation", "Not meet Thai regulation"])

    with col2:
        airpurifier = st.selectbox("Air Purifier", ["Off", "On"])
        outdoor_temp = st.selectbox("Outdoor Temperature (°C)", ["≤29", ">29"])
        outdoor_pm = st.selectbox("Outdoor PM2.5 (µg/m³)", ["≤25", ">25"])
        outdoor_rh = st.selectbox("Outdoor Relative Humidity (%)", ["≤70", ">70"])

    submitted = st.form_submit_button("🔎 Predict IAQ")

# --- Helper Functions ---
def binary(value, true_value):
    return 1 if value == true_value else 0

# Activity-related PM2.5 emission levels
activity_pm25_map = {
    "Sleeping": 1,
    "Sitting": 1,
    "Watching TV": 1,
    "Working": 1,
    "Having Dinner": 1,
    "Dressing": 1,
    "Cleaning": 1,
    "Cooking": 2,
    "Smoking": 2
}
activity_pm25 = activity_pm25_map[activity]

# --- Feature Vectors ---
def get_features_temp():
    return [[
        binary(outdoor_temp, "≤29"),
        binary(window, "Open"),
        binary(ac, "Off")
    ]]

def get_features_rh():
    return [[
        binary(outdoor_rh, ">70"),
        binary(window, "Open"),
        binary(ac, "On")
    ]]

def get_features_co2():
    return [[
        occupancy,
        binary(window, "Open"),
        binary(ach, "Meet Thai regulation")
    ]]

def get_features_pm25():
    return [[
        binary(outdoor_pm, ">25"),
        binary(window, "Open"),
        activity_pm25,
        binary(airpurifier, "On")
    ]]

# --- Prediction ---
def predict(model_path, features):
    model = joblib.load(model_path)
    prediction = model.predict(features)[0]
    exceed_prob = model.predict_proba(features)[0][1]
    return prediction, exceed_prob

def show_result(name, prediction, exceed_prob):
    emoji = "⚠️ Exceeded" if prediction else "✅ Within Limit"
    color = "red" if prediction else "green"
    percent = f"{exceed_prob * 100:.1f}%"
    st.markdown(
        f"<b style='color:{color}'>{name}:</b> {emoji} "
        f"(<i>{percent} chance to exceed</i>)",
        unsafe_allow_html=True
    )

# --- Run Predictions ---
if submitted:
    st.subheader("🔍 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        temp_pred, temp_prob = predict("model_Temp.pkl", get_features_temp())
        rh_pred, rh_prob = predict("model_RH.pkl", get_features_rh())
    with col2:
        pm25_pred, pm25_prob = predict("model_PM25.pkl", get_features_pm25())
        co2_pred, co2_prob = predict("model_CO2.pkl", get_features_co2())

    show_result("🌡️ Temperature", temp_pred, temp_prob)
    show_result("💧 Relative Humidity", rh_pred, rh_prob)
    show_result("🫁 PM2.5", pm25_pred, pm25_prob)
    show_result("😮‍💨 CO2", co2_pred, co2_prob)
