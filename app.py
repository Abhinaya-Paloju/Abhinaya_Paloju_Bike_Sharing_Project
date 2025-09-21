import streamlit as st
from xgboost import XGBRegressor
import json
import pandas as pd
import numpy as np

# ---------------------------
# Load trained model + features
# ---------------------------

FEATURES_PATH = "artifacts/features.json"
model = XGBRegressor()
model.load_model("artifacts/model.json")
with open(FEATURES_PATH, "r") as f:
    features = json.load(f)["features"]

st.title("🚴 Bike Sharing Demand Prediction")

st.write("Move the sliders / choose values to predict the number of bike rentals (cnt).")

# ---------------------------
# User Inputs
# ---------------------------
season = st.selectbox("Season (1=Winter, 2=Spring, 3=Summer, 4=Fall)", [1, 2, 3, 4])
weathersit = st.selectbox("Weather Situation (1=Clear, 2=Mist, 3=Light Snow/Rain, 4=Heavy Rain)", [1, 2, 3, 4])
temp = st.slider("Temperature (normalized 0–1)", 0.0, 1.0, 0.3)
hum = st.slider("Humidity (normalized 0–1)", 0.0, 1.0, 0.55)
windspeed = st.slider("Windspeed (normalized 0–1)", 0.0, 1.0, 0.12)

hour = st.slider("Hour of Day", 0, 23, 14)
day_of_week = st.slider("Day of Week (0=Mon)", 0, 6, 3)
month = st.slider("Month", 1, 12, 5)
is_weekend = st.selectbox("Is Weekend?", [0, 1])

# ---------------------------
# Feature Engineering (auto-compute)
# ---------------------------
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
dow_sin = np.sin(2 * np.pi * day_of_week / 7)
dow_cos = np.cos(2 * np.pi * day_of_week / 7)
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)

# Lag / rolling features (for demo: approximate values)
cnt_lag1 = st.number_input("Previous Hour Count (cnt_lag1)", value=200)
cnt_lag24 = st.number_input("Previous Day Count (cnt_lag24)", value=250)
cnt_roll3 = st.number_input("Rolling 3-hr Avg", value=230)
cnt_roll6 = st.number_input("Rolling 6-hr Avg", value=220)
cnt_roll12 = st.number_input("Rolling 12-hr Avg", value=210)

# Interaction features
hour_temp = hour * temp
hour_humidity = hour * hum
temp_humidity = temp * hum

# ---------------------------
# Prepare Data
# ---------------------------
input_data = {
    "season": season,
    "weathersit": weathersit,
    "temp": temp,
    "hum": hum,
    "windspeed": windspeed,
    "hour": hour,
    "day_of_week": day_of_week,
    "month": month,
    "is_weekend": is_weekend,
    "hour_sin": hour_sin,
    "hour_cos": hour_cos,
    "dow_sin": dow_sin,
    "dow_cos": dow_cos,
    "month_sin": month_sin,
    "month_cos": month_cos,
    "cnt_lag1": cnt_lag1,
    "cnt_lag24": cnt_lag24,
    "cnt_roll3": cnt_roll3,
    "cnt_roll6": cnt_roll6,
    "cnt_roll12": cnt_roll12,
    "hour_temp": hour_temp,
    "hour_humidity": hour_humidity,
    "temp_humidity": temp_humidity
}

df = pd.DataFrame([input_data]).reindex(columns=features)

# ---------------------------
# Predict Button
# ---------------------------
if st.button("🔮 Predict Bike Demand"):
    prediction = model.predict(df)[0]
    st.success(f"Predicted Bike Count: {prediction:.0f}")

