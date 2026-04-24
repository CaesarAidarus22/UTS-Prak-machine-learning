import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
API_URL = f"{BACKEND_URL}/predict"
HEALTH_URL = f"{BACKEND_URL}/health"
REQUEST_TIMEOUT = 30

FIELD_LIMITS = {
    "bulk_density": (0.0, 3.0, 1.2, 0.1),
    "organic_matter_pct": (0.0, 100.0, 3.0, 0.1),
    "cation_exchange_capacity": (0.0, 100.0, 15.0, 0.1),
    "salinity_ec": (0.0, 50.0, 0.5, 0.1),
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

st.markdown(
    """
    <style>
    :root {
        --text-main: #0f172a;
        --text-muted: #64748b;
        --bg-app: #f8fafc;
        --brand: #065f46;
        --brand-dark: #103629;
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
        max-width: 1160px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: var(--brand) !important;
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
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        font-size: 1.7rem !important;
        font-weight: 800 !important;
        color: #f8fafc !important;
        margin-top: 0.1rem !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.08) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.22) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .stButton > button * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.14) !important;
        color: #ffffff !important;
        border-color: rgba(255,255,255,0.34) !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover * {
        color: #ffffff !important;
        fill: #ffffff !important;
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
        max-width: 720px;
    }
    [data-testid="stForm"] {
        border-radius: 18px;
        border: 1px solid var(--line);
        background-color: var(--card);
        padding: 2rem;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.05);
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
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border-color: #cbd5e1;
        color: var(--text-main);
    }
    div[data-baseweb="input"] > div {
        border-radius: 10px;
        border: 1px solid #cbd5e1;
        background-color: #ffffff;
    }
    div[data-baseweb="input"] input {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }
    div[data-baseweb="input"] input::placeholder {
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
    [data-testid="stWidgetLabel"] p,
    [data-testid="stMarkdownContainer"] p {
        color: #0f172a !important;
        opacity: 1 !important;
    }
    [data-testid="stWidgetLabel"] p {
        font-weight: 700 !important;
    }
    div[data-baseweb="input"] > div:focus-within {
        border-color: #16a34a;
        box-shadow: 0 0 0 1px #16a34a;
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
    .status-badge::before {
        content: "";
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
        background: currentColor;
    }
    .ready { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
    .warning { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
    .offline { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
    [data-testid="stSidebar"] .status-badge.ready {
        background: #dcfce7 !important;
        color: #14532d !important;
        border: 1px solid #bbf7d0 !important;
    }
    [data-testid="stSidebar"] .status-badge.warning {
        background: #fef3c7 !important;
        color: #92400e !important;
        border: 1px solid #fde68a !important;
    }
    [data-testid="stSidebar"] .status-badge.offline {
        background: #fee2e2 !important;
        color: #991b1b !important;
        border: 1px solid #fecaca !important;
    }
    [data-testid="stSidebar"] .status-badge.ready::before,
    [data-testid="stSidebar"] .status-badge.warning::before,
    [data-testid="stSidebar"] .status-badge.offline::before {
        background: currentColor !important;
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
    .result-success {
        background: #f0fdf4;
        border-color: #16a34a;
        color: #14532d;
    }
    .result-danger {
        background: #fef2f2;
        border-color: #dc2626;
        color: #7f1d1d;
    }
    .result-banner h3 {
        margin: 0 0 0.45rem 0 !important;
        font-size: 1.5rem;
        color: inherit !important;
    }
    .result-banner p {
        margin: 0;
        font-size: 1rem;
        color: inherit;
        opacity: 0.96;
    }
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
        font-size: 1.25rem;
        color: #0f172a;
        font-weight: 800;
        line-height: 1.4;
        word-break: break-word;
    }
    .subtitle {
        color: #64748b !important;
        margin-bottom: 1rem !important;
        font-size: 1rem;
        line-height: 1.7;
    }
    @media (max-width: 900px) {
        .hero-title {
            font-size: 2.3rem;
        }
        .result-grid {
            grid-template-columns: 1fr;
        }
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
            Sistem prediksi kelayakan kondisi tanah secara cepat dan akurat. Antarmuka ini dirancang
            agar mudah dibaca, rapi saat presentasi, dan jelas saat menampilkan hasil prediksi.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

main_col, side_col = st.columns([2, 1], gap="large")

with main_col:
    st.subheader("Parameter Tanah")
    st.markdown(
        "<p class='subtitle'>Masukkan metrik tanah dalam rentang yang diizinkan untuk memulai prediksi.</p>",
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        form_left, form_right = st.columns(2, gap="medium")

        with form_left:
            bulk_density = st.number_input("Bulk Density", *FIELD_LIMITS["bulk_density"])
            organic_matter_pct = st.number_input("Organic Matter (%)", *FIELD_LIMITS["organic_matter_pct"])

        with form_right:
            cation_exchange_capacity = st.number_input(
                "Cation Exchange Capacity",
                *FIELD_LIMITS["cation_exchange_capacity"],
            )
            salinity_ec = st.number_input("Salinity EC", *FIELD_LIMITS["salinity_ec"])

        submit = st.form_submit_button("Jalankan Prediksi", use_container_width=True)

with side_col:
    st.subheader("System Status")
    st.markdown(
        "<p class='subtitle'>Ringkasan kesiapan model dan sistem.</p>",
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
            """.format(
                model_status="Loaded" if health_data.get("model_loaded") else "Not Ready",
                dataset_status="Available" if health_data.get("dataset_exists") else "Missing",
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
                    "Kondisi tanah ini diprediksi sangat baik dan memenuhi standar untuk mendukung pertumbuhan tanaman."
                    if prediction == 0
                    else "Kondisi tanah ini kurang ideal. Diperlukan penanganan lebih lanjut sebelum digunakan untuk penanaman."
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
