
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
CROP_MODEL = BASE / "models" / "crop_model.joblib"
FERT_MODEL = BASE / "models" / "fertilizer_model.joblib"

st.set_page_config(
    page_title="AgriAI - Crop & Fertilizer Recommendation",
    page_icon="🌱",
    layout="wide"
)

@st.cache_resource
def load_models():
    crop = joblib.load(CROP_MODEL)
    fertilizer = joblib.load(FERT_MODEL)
    return crop, fertilizer

crop_model, fertilizer_model = load_models()

st.title("🌱 AgriAI — Crop & Fertilizer Recommendation System")
st.caption("Machine-learning based agricultural recommendation system")

tab1, tab2, tab3 = st.tabs([
    "🌾 Crop Recommendation",
    "🧪 Fertilizer Recommendation",
    "📊 Model Information"
])

with tab1:
    st.subheader("Recommend the best crop")
    st.write("Enter the soil and climate conditions.")

    c1, c2, c3 = st.columns(3)
    with c1:
        n = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=90.0)
        p = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=42.0)
        k = st.number_input("Potassium (K)", min_value=0.0, max_value=250.0, value=43.0)
    with c2:
        temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0)
    with c3:
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=5000.0, value=200.0)

    if st.button("🌾 Predict Best Crop", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "N": n, "P": p, "K": k,
            "temperature": temperature, "humidity": humidity,
            "ph": ph, "rainfall": rainfall
        }])
        prediction = crop_model.predict(row)[0]

        st.success(f"Recommended Crop: **{prediction}**")

        if hasattr(crop_model, "predict_proba"):
            probs = crop_model.predict_proba(row)[0]
            classes = crop_model.classes_
            top = np.argsort(probs)[::-1][:5]
            chart = pd.DataFrame({
                "Crop": [classes[i] for i in top],
                "Probability": [float(probs[i]) for i in top]
            }).set_index("Crop")
            st.write("Top model predictions")
            st.bar_chart(chart)

with tab2:
    st.subheader("Recommend the best fertilizer")

    c1, c2, c3 = st.columns(3)
    with c1:
        temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=26.0, key="ft")
        hum = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=52.0, key="fh")
        moisture = st.number_input("Moisture (%)", min_value=0.0, max_value=100.0, value=38.0)
    with c2:
        soil = st.selectbox("Soil Type", [
            "Sandy", "Loamy", "Black", "Red", "Clayey"
        ])
        crop = st.selectbox("Crop Type", [
            "Maize", "Sugarcane", "Cotton", "Tobacco", "Paddy",
            "Barley", "Wheat", "Millets", "Oil seeds", "Pulses",
            "Ground Nuts"
        ])
    with c3:
        fn = st.number_input("Nitrogen", min_value=0.0, max_value=200.0, value=37.0)
        fk = st.number_input("Potassium", min_value=0.0, max_value=200.0, value=0.0)
        fp = st.number_input("Phosphorous", min_value=0.0, max_value=200.0, value=0.0)

    if st.button("🧪 Predict Fertilizer", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "Temparature": temp,
            "Humidity": hum,
            "Moisture": moisture,
            "Soil Type": soil,
            "Crop Type": crop,
            "Nitrogen": fn,
            "Potassium": fk,
            "Phosphorous": fp
        }])
        prediction = fertilizer_model.predict(row)[0]
        st.success(f"Recommended Fertilizer: **{prediction}**")

        if hasattr(fertilizer_model, "predict_proba"):
            probs = fertilizer_model.predict_proba(row)[0]
            classes = fertilizer_model.classes_
            top = np.argsort(probs)[::-1][:5]
            chart = pd.DataFrame({
                "Fertilizer": [classes[i] for i in top],
                "Probability": [float(probs[i]) for i in top]
            }).set_index("Fertilizer")
            st.write("Top model predictions")
            st.bar_chart(chart)

with tab3:
    st.subheader("Machine Learning Pipeline")

    st.markdown("""
    **This is a genuine end-to-end ML project:**

    1. Real agricultural datasets are used.
    2. Data is split into training and testing sets.
    3. Features are preprocessed using `StandardScaler` and `OneHotEncoder`.
    4. `RandomForestClassifier` is trained on the training data.
    5. The trained pipelines are saved as model artifacts.
    6. The Streamlit application loads those trained models.
    7. User input is passed through the same preprocessing pipeline.
    8. The ML model predicts the crop/fertilizer.
    """)

    for filename, title in [
        ("reports/crop_metrics.json", "Crop Model"),
        ("reports/fertilizer_metrics.json", "Fertilizer Model")
    ]:
        path = BASE / filename
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            st.write(f"### {title}")
            st.metric("Test Accuracy", f"{data['accuracy'] * 100:.2f}%")
            st.write("Algorithm:", data["algorithm"])
            st.write("Training samples:", data["samples"])

st.divider()
st.caption("AgriAI • Educational ML project • Predictions are model-based and should not replace agronomist advice.")
