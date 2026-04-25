import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
API_URL = f"{BACKEND_URL}/predict"
HEALTH_URL = f"{BACKEND_URL}/health"
REQUEST_TIMEOUT = 30

NUMERIC_FIELDS = {
    "bulk_density": {"label": "Bulk Density\n(Kepadatan Tanah)", "min_value": 0.7, "max_value": 1.6, "value": 1.5, "step": 0.1},
    "organic_matter_pct": {"label": "Organic Matter (%)\n(Bahan Organik)", "min_value": 1.2, "max_value": 18.0, "value": 2.0, "step": 0.1},
    "cation_exchange_capacity": {"label": "Cation Exchange Capacity\n(Kapasitas Tukar Kation)", "min_value": 5.0, "max_value": 40.0, "value": 8.0, "step": 0.1},
    "salinity_ec": {"label": "Salinity EC\n(Salinitas Tanah)", "min_value": 0.2, "max_value": 4.0, "value": 0.3, "step": 0.1},
    "buffering_capacity": {"label": "Buffering Capacity\n(Kapasitas Penyangga Tanah)", "min_value": 0.3, "max_value": 0.9, "value": 0.4, "step": 0.1},
    "soil_moisture_pct": {"label": "Soil Moisture (%)\n(Kelembapan Tanah)", "min_value": 5.0, "max_value": 70.0, "value": 44.9, "step": 0.1},
    "moisture_limit_dry": {"label": "Moisture Limit Dry\n(Batas Kering Kelembapan)", "min_value": 8.0, "max_value": 30.0, "value": 12.0, "step": 0.1},
    "moisture_limit_wet": {"label": "Moisture Limit Wet\n(Batas Basah Kelembapan)", "min_value": 28.0, "max_value": 65.0, "value": 35.0, "step": 0.1},
    "soil_temp_c": {"label": "Soil Temperature (C)\n(Suhu Tanah)", "min_value": 10.0, "max_value": 40.0, "value": 30.3, "step": 0.1},
    "air_temp_c": {"label": "Air Temperature (C)\n(Suhu Udara)", "min_value": 7.0, "max_value": 49.6, "value": 34.1, "step": 0.1},
    "light_intensity_par": {"label": "Light Intensity PAR\n(Intensitas Cahaya PAR)", "min_value": 200.0, "max_value": 1200.0, "value": 408.0, "step": 1.0},
    "soil_ph": {"label": "Soil pH\n(pH Tanah)", "min_value": 4.0, "max_value": 8.8, "value": 5.01, "step": 0.1},
    "ph_stress_flag": {"label": "pH Stress Flag\n(Penanda Stres pH)", "min_value": 0, "max_value": 1, "value": 1, "step": 1},
    "nitrogen_ppm": {"label": "Nitrogen (ppm)\n(Nitrogen)", "min_value": 20.0, "max_value": 220.0, "value": 84.5, "step": 0.1},
    "phosphorus_ppm": {"label": "Phosphorus (ppm)\n(Fosfor)", "min_value": 10.0, "max_value": 159.0, "value": 83.6, "step": 0.1},
    "potassium_ppm": {"label": "Potassium (ppm)\n(Kalium)", "min_value": 20.0, "max_value": 220.0, "value": 83.6, "step": 0.1},
}
CATEGORICAL_FIELDS = {
    "soil_type": {
        "label": "Soil Type\n(Jenis Tanah)",
        "options": ["Alluvial", "Chalky", "Clayey", "Laterite", "Loamy", "Peaty", "Saline", "Sandy", "Silty"],
        "default": "Chalky",
    },
    "moisture_regime": {
        "label": "Moisture Regime\n(Kondisi Kelembapan)",
        "options": ["dry", "optimal", "waterlogged"],
        "default": "waterlogged",
    },
    "thermal_regime": {
        "label": "Thermal Regime\n(Kondisi Suhu)",
        "options": ["cold", "heat_stress", "optimal"],
        "default": "heat_stress",
    },
    "nutrient_balance": {
        "label": "Nutrient Balance\n(Keseimbangan Nutrisi)",
        "options": ["deficient", "excessive", "optimal"],
        "default": "deficient",
    },
    "plant_category": {
        "label": "Plant Category\n(Kategori Tanaman)",
        "options": ["cereal", "legume", "vegetable"],
        "default": "vegetable",
    },
}


def fetch_health() -> dict | None:
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def extract_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
        return payload.get("detail") or payload.get("error") or "Unknown error"
    except ValueError:
        return response.text or "Unknown error"


def status_variant(health_data: dict | None) -> tuple[str, str]:
    if not health_data:
        return "offline", "Backend offline"
    if health_data.get("model_loaded", False):
        return "ready", "Backend siap digunakan"
    return "warning", "Backend aktif, model belum siap"


st.set_page_config(page_title="Agro ML Predictor", page_icon="🌿", layout="wide")

def numeric_input_config(field_name: str) -> dict:
    field_config = dict(NUMERIC_FIELDS[field_name])
    min_value = field_config.pop("min_value", field_config.pop("min", None))
    max_value = field_config.pop("max_value", field_config.pop("max", None))
    value = field_config.pop("value", field_config.pop("default", None))

    if min_value is not None:
        field_config["min_value"] = min_value
    if max_value is not None:
        field_config["max_value"] = max_value
    if value is not None:
        field_config["value"] = value

    field_config.setdefault("key", field_name)
    if min_value is not None and max_value is not None:
        field_config.setdefault("help", f"Rentang input: {min_value} - {max_value}")

    if field_config.get("step") == 1 and all(
        isinstance(number, int) for number in (min_value, max_value, value) if number is not None
    ):
        field_config.setdefault("format", "%d")

    return field_config


st.markdown(
    """
    <style>
    :root {
        --text-main: #0f172a;
        --text-muted: #64748b;
        --bg-app: #f8fafc;
        --brand: #065f46;
        --sidebar: #182b22;
        --sidebar-soft: #21382d;
        --card: #ffffff;
        --line: #e2e8f0;
    }
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #f4f7fb 100%);
        color: var(--text-main);
    }
    .block-container {
        max-width: 1240px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: var(--brand) !important;
        font-weight: 800 !important;
    }
    h4 {
        color: #0f5132 !important;
        font-weight: 800 !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--sidebar) 0%, var(--sidebar-soft) 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] * {
        color: #f8fafc;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: #f8fafc !important;
    }
    [data-testid="stSidebar"] code {
        color: #bbf7d0 !important;
        background: rgba(255,255,255,0.08) !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.08) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.22) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.14) !important;
        color: #ffffff !important;
        border-color: rgba(255,255,255,0.34) !important;
    }
    .hero-card {
        background:
            radial-gradient(circle at top right, rgba(22, 163, 74, 0.08), transparent 22%),
            linear-gradient(135deg, #ffffff 0%, #f7fffb 100%);
        border: 1px solid #dbe7df;
        border-radius: 28px;
        padding: 2.2rem 2rem;
        box-shadow: 0 20px 45px rgba(15, 23, 42, 0.07);
        text-align: center;
        margin-bottom: 1.8rem;
    }
    .hero-kicker {
        display: inline-block;
        padding: 0.38rem 0.8rem;
        border-radius: 999px;
        background: #ecfdf5;
        border: 1px solid #bbf7d0;
        color: #166534;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .hero-title {
        margin: 0;
        font-size: 3.1rem;
        line-height: 1.05;
        color: var(--brand) !important;
        font-weight: 800;
    }
    .hero-text {
        margin: 0.9rem auto 0 auto;
        color: var(--text-muted) !important;
        font-size: 1.08rem;
        line-height: 1.75;
        max-width: 760px;
    }
    .subtitle {
        color: #64748b !important;
        margin-bottom: 1rem !important;
        font-size: 1rem;
        line-height: 1.7;
    }
    [data-testid="stForm"] {
        border-radius: 18px;
        border: 1px solid var(--line);
        background-color: var(--card);
        padding: 2rem;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.05);
    }
    [data-testid="stForm"] h4 {
        color: #0f5132 !important;
        margin-top: 0.35rem !important;
        margin-bottom: 1rem !important;
        letter-spacing: -0.01em;
    }
    [data-testid="stFormSubmitButton"] > button {
        background-color: #16a34a !important;
        color: #ffffff !important;
        border-radius: 10px;
        border: none;
        font-weight: 700;
        font-size: 1.05rem;
        min-height: 3.2rem;
    }
    [data-testid="stFormSubmitButton"] > button:hover {
        background-color: #15803d !important;
        color: #ffffff !important;
        box-shadow: 0 6px 16px rgba(22, 163, 74, 0.24);
    }
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        border-radius: 10px;
        border: 1px solid #cbd5e1;
        background-color: #ffffff;
    }
    div[data-baseweb="input"] input,
    div[data-baseweb="select"] input {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }
    div[data-baseweb="select"] div,
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] svg {
        color: #0f172a !important;
        fill: #0f172a !important;
        opacity: 1 !important;
    }
    .stSelectbox div[data-baseweb="select"] > div {
        color: #0f172a !important;
        background-color: #ffffff !important;
    }
    .stSelectbox div[data-baseweb="select"] [class*="singleValue"],
    .stSelectbox div[data-baseweb="select"] [class*="placeholder"] {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        opacity: 1 !important;
    }
    div[data-baseweb="input"] input::placeholder,
    div[data-baseweb="select"] input::placeholder {
        color: #475569 !important;
        -webkit-text-fill-color: #475569 !important;
        opacity: 1 !important;
    }
    button[aria-label="Increment value"],
    button[aria-label="Decrement value"] {
        color: #ffffff !important;
        opacity: 1 !important;
    }
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p {
        color: #0f172a !important;
        opacity: 1 !important;
    }
    [data-testid="stWidgetLabel"] p {
        font-weight: 700 !important;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.45rem 1rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 1rem;
    }
    [data-testid="stSidebar"] .status-badge {
        width: 100%;
        justify-content: center;
        padding: 0.7rem 1rem;
        font-size: 0.95rem;
        font-weight: 900;
        letter-spacing: 0.04em;
        box-shadow: 0 8px 18px rgba(0,0,0,0.14);
    }
    .status-badge::before {
        content: "";
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
        background: currentColor;
    }
    [data-testid="stSidebar"] .status-badge::before {
        width: 10px;
        height: 10px;
        flex: 0 0 10px;
    }
    .ready { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
    .warning { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
    .offline { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
    [data-testid="stSidebar"] .status-badge.ready {
        background: #dcfce7 !important;
        color: #064e3b !important;
        border: 1px solid #86efac !important;
    }
    [data-testid="stSidebar"] .status-badge.warning {
        background: #fef3c7 !important;
        color: #78350f !important;
        border: 1px solid #fcd34d !important;
    }
    [data-testid="stSidebar"] .status-badge.offline {
        background: #fee2e2 !important;
        color: #7f1d1d !important;
        border: 1px solid #fca5a5 !important;
    }
    .info-card {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1.2rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
    }
    .info-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .info-val {
        font-size: 1.08rem;
        color: #0f172a;
        font-weight: 700;
    }
    .result-banner {
        padding: 1.5rem;
        border-radius: 14px;
        margin: 1rem 0 1.2rem 0;
        border-left: 6px solid;
    }
    .result-success { background: #f0fdf4; border-color: #16a34a; color: #14532d; }
    .result-danger { background: #fef2f2; border-color: #dc2626; color: #7f1d1d; }
    .result-banner h3 { margin: 0 0 0.45rem 0 !important; font-size: 1.5rem; color: inherit !important; }
    .result-banner p { margin: 0; font-size: 1rem; color: inherit; opacity: 0.96; }
    .result-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .result-card {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .result-card-label {
        font-size: 0.78rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .result-card-value {
        font-size: 1.18rem;
        color: #0f172a;
        font-weight: 800;
        line-height: 1.4;
        word-break: break-word;
    }
    @media (max-width: 980px) {
        .hero-title { font-size: 2.4rem; }
        .result-grid { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "health_data" not in st.session_state:
    st.session_state.health_data = fetch_health()

state_class, state_label = status_variant(st.session_state.health_data)

with st.sidebar:
    st.markdown("### Control Panel")
    st.caption(f"API Base URL: `{BACKEND_URL}`")
    if st.button("Refresh Status", use_container_width=True):
        st.session_state.health_data = fetch_health()
        state_class, state_label = status_variant(st.session_state.health_data)

    st.markdown(
        f"<div class='status-badge {state_class}' style='margin-top: 1rem;'>{state_label}</div>",
        unsafe_allow_html=True,
    )
    if not st.session_state.health_data:
        st.info("Jalankan backend terlebih dahulu sebelum melakukan prediksi.")

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-kicker">Agro ML System</div>
        <h1 class="hero-title">Agro ML Predictor</h1>
        <p class="hero-text">
            Sistem prediksi kelayakan kondisi tanah berbasis fitur numerik dan kategorikal yang lebih lengkap,
            agar hasil prediksi lebih informatif dan lebih mendekati kebutuhan analisis pada instruksi UTS.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

main_col, side_col = st.columns([2.1, 1], gap="large")

with main_col:
    st.subheader("Parameter Tanah")
    st.markdown(
        "<p class='subtitle'>Masukkan parameter agro-environmental yang lebih lengkap untuk menghasilkan prediksi yang lebih kaya.</p>",
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        st.markdown("#### Properti Dasar Tanah")
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            bulk_density = st.number_input(**numeric_input_config("bulk_density"))
            organic_matter_pct = st.number_input(**numeric_input_config("organic_matter_pct"))
            cation_exchange_capacity = st.number_input(**numeric_input_config("cation_exchange_capacity"))
            salinity_ec = st.number_input(**numeric_input_config("salinity_ec"))
        with col2:
            buffering_capacity = st.number_input(**numeric_input_config("buffering_capacity"))
            soil_moisture_pct = st.number_input(**numeric_input_config("soil_moisture_pct"))
            moisture_limit_dry = st.number_input(**numeric_input_config("moisture_limit_dry"))
            moisture_limit_wet = st.number_input(**numeric_input_config("moisture_limit_wet"))

        st.markdown("#### Suhu, Cahaya, dan Kimia Tanah")
        col3, col4 = st.columns(2, gap="medium")
        with col3:
            soil_temp_c = st.number_input(**numeric_input_config("soil_temp_c"))
            air_temp_c = st.number_input(**numeric_input_config("air_temp_c"))
            light_intensity_par = st.number_input(**numeric_input_config("light_intensity_par"))
            soil_ph = st.number_input(**numeric_input_config("soil_ph"))
        with col4:
            ph_stress_flag = st.number_input(**numeric_input_config("ph_stress_flag"))
            nitrogen_ppm = st.number_input(**numeric_input_config("nitrogen_ppm"))
            phosphorus_ppm = st.number_input(**numeric_input_config("phosphorus_ppm"))
            potassium_ppm = st.number_input(**numeric_input_config("potassium_ppm"))

        st.markdown("#### Konteks Kategorikal")
        col5, col6 = st.columns(2, gap="medium")
        with col5:
            soil_type = st.selectbox(
                CATEGORICAL_FIELDS["soil_type"]["label"],
                CATEGORICAL_FIELDS["soil_type"]["options"],
                index=CATEGORICAL_FIELDS["soil_type"]["options"].index(CATEGORICAL_FIELDS["soil_type"]["default"]),
            )
            moisture_regime = st.selectbox(
                CATEGORICAL_FIELDS["moisture_regime"]["label"],
                CATEGORICAL_FIELDS["moisture_regime"]["options"],
                index=CATEGORICAL_FIELDS["moisture_regime"]["options"].index(CATEGORICAL_FIELDS["moisture_regime"]["default"]),
            )
            thermal_regime = st.selectbox(
                CATEGORICAL_FIELDS["thermal_regime"]["label"],
                CATEGORICAL_FIELDS["thermal_regime"]["options"],
                index=CATEGORICAL_FIELDS["thermal_regime"]["options"].index(CATEGORICAL_FIELDS["thermal_regime"]["default"]),
            )
        with col6:
            nutrient_balance = st.selectbox(
                CATEGORICAL_FIELDS["nutrient_balance"]["label"],
                CATEGORICAL_FIELDS["nutrient_balance"]["options"],
                index=CATEGORICAL_FIELDS["nutrient_balance"]["options"].index(CATEGORICAL_FIELDS["nutrient_balance"]["default"]),
            )
            plant_category = st.selectbox(
                CATEGORICAL_FIELDS["plant_category"]["label"],
                CATEGORICAL_FIELDS["plant_category"]["options"],
                index=CATEGORICAL_FIELDS["plant_category"]["options"].index(CATEGORICAL_FIELDS["plant_category"]["default"]),
            )

        submit = st.form_submit_button("Jalankan Prediksi", use_container_width=True)

with side_col:
    st.subheader("System Status")
    st.markdown(
        "<p class='subtitle'>Ringkasan kesiapan model dan jumlah fitur yang dipakai oleh sistem.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='status-badge {state_class}'>{state_label}</div>",
        unsafe_allow_html=True,
    )
    if st.session_state.health_data:
        health_data = st.session_state.health_data
        st.markdown(
            """
            <div class="info-card">
                <div class="info-label">Model</div>
                <div class="info-val">{model_status}</div>
            </div>
            <div class="info-card">
                <div class="info-label">Dataset</div>
                <div class="info-val">{dataset_status}</div>
            </div>
            <div class="info-card">
                <div class="info-label">Feature Count</div>
                <div class="info-val">{feature_count} fitur</div>
            </div>
            """.format(
                model_status="Loaded" if health_data.get("model_loaded") else "Not Ready",
                dataset_status="Available" if health_data.get("dataset_exists") else "Missing",
                feature_count=len(health_data.get("features", [])),
            ),
            unsafe_allow_html=True,
        )
    else:
        st.warning("Data kesehatan backend belum tersedia.")

if submit:
    payload = {
        "bulk_density": bulk_density,
        "organic_matter_pct": organic_matter_pct,
        "cation_exchange_capacity": cation_exchange_capacity,
        "salinity_ec": salinity_ec,
        "buffering_capacity": buffering_capacity,
        "soil_moisture_pct": soil_moisture_pct,
        "moisture_limit_dry": moisture_limit_dry,
        "moisture_limit_wet": moisture_limit_wet,
        "soil_temp_c": soil_temp_c,
        "air_temp_c": air_temp_c,
        "light_intensity_par": light_intensity_par,
        "soil_ph": soil_ph,
        "ph_stress_flag": int(ph_stress_flag),
        "nitrogen_ppm": nitrogen_ppm,
        "phosphorus_ppm": phosphorus_ppm,
        "potassium_ppm": potassium_ppm,
        "soil_type": soil_type,
        "moisture_regime": moisture_regime,
        "thermal_regime": thermal_regime,
        "nutrient_balance": nutrient_balance,
        "plant_category": plant_category,
    }

    with st.spinner("Menjalankan prediksi..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
            if response.ok:
                result = response.json()
                prediction = result.get("prediction")
                confidence = result.get("confidence")
                banner_class = "result-success" if prediction == 0 else "result-danger"
                banner_title = (
                    "Kondisi Tanah: Suitable"
                    if prediction == 0
                    else "Kondisi Tanah: Not Suitable"
                )
                banner_copy = (
                    "Kondisi tanah ini diprediksi mendukung pertumbuhan tanaman berdasarkan kombinasi fitur tanah, kelembapan, suhu, nutrien, dan konteks kategorikal."
                    if prediction == 0
                    else "Kondisi tanah ini diprediksi kurang ideal berdasarkan kombinasi fitur tanah, kelembapan, suhu, nutrien, dan konteks kategorikal."
                )

                st.markdown("---")
                st.subheader("Hasil Prediksi")
                st.markdown(
                    f"""
                    <div class="result-banner {banner_class}">
                        <h3>{banner_title}</h3>
                        <p>{banner_copy}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div class="result-grid">
                        <div class="result-card">
                            <div class="result-card-label">Predicted Class</div>
                            <div class="result-card-value">{prediction}</div>
                        </div>
                        <div class="result-card">
                            <div class="result-card-label">Confidence</div>
                            <div class="result-card-value">{f"{confidence * 100:.2f}%" if confidence is not None else "N/A"}</div>
                        </div>
                        <div class="result-card">
                            <div class="result-card-label">Interpretation</div>
                            <div class="result-card-value">{result.get("interpretation", "-")}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.expander("Lihat payload"):
                    st.json(payload)
            else:
                st.error(extract_error_message(response))
        except requests.RequestException as exc:
            st.error(f"Error saat menghubungi backend: {exc}")
