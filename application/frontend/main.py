import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Agro ML Predictor", layout="centered")

st.title("🌱 Agro-Environmental Suitability Predictor")

st.write("Masukkan kondisi tanah untuk prediksi.")

# =========================
# Input Form
# =========================
with st.form("prediction_form"):

    st.subheader("Soil Properties")

    bulk_density = st.number_input("Bulk Density", 0.0, 3.0, 1.2)
    organic_matter_pct = st.number_input("Organic Matter (%)", 0.0, 100.0, 3.0)
    cec = st.number_input("Cation Exchange Capacity", 0.0, 100.0, 15.0)
    salinity_ec = st.number_input("Salinity EC", 0.0, 50.0, 0.5)

    submit = st.form_submit_button("Predict")

# =========================
# Handle Prediction
# =========================
if submit:
    payload = {
        "bulk_density": bulk_density,
        "organic_matter_pct": organic_matter_pct,
        "cation_exchange_capacity": cec,
        "salinity_ec": salinity_ec
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()

            if "prediction" in result:
                if result["prediction"] == 0:
                    st.success("✅ Tanaman bisa tumbuh (Suitable)")
                else:
                    st.error("❌ Tanaman tidak cocok (Not Suitable)")
            else:
                st.error(result.get("error", "Unknown error"))

        else:
            st.error("Gagal menghubungi backend")

    except Exception as e:
        st.error(f"Error: {e}")