import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="FinCommerce Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paleta FinCommerce ───────────────────────────────────────────────────────
# Colores principales:
# Navy: #001B3A | Blue: #003566 | Aqua: #11D6BE | Aqua hover: #00F0C8
# Background: #F5F7FA | Muted text: #6B778C | Dark panel: #0D1117

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    :root {
        --fincommerce-navy: #001B3A;
        --fincommerce-blue: #003566;
        --fincommerce-aqua: #11D6BE;
        --fincommerce-aqua-hover: #00F0C8;
        --fincommerce-bg: #F5F7FA;
        --fincommerce-text-muted: #6B778C;
        --fincommerce-dark-panel: #0D1117;
        --fincommerce-card: #FFFFFF;
        --fincommerce-border: rgba(0, 27, 58, 0.08);
        --fincommerce-alert: #FF5252;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: var(--fincommerce-bg) !important;
        color: var(--fincommerce-navy) !important;
    }

    header[data-testid="stHeader"] {
        background-color: var(--fincommerce-dark-panel) !important;
    }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1100px !important;
    }

    h1, h2, h3 {
        color: var(--fincommerce-navy) !important;
        font-weight: 800;
        letter-spacing: -0.02em;
    }

    p, li, .stCaptionContainer {
        color: var(--fincommerce-text-muted) !important;
    }

    .hero {
        background: linear-gradient(135deg, var(--fincommerce-navy) 0%, var(--fincommerce-blue) 72%, var(--fincommerce-dark-panel) 100%);
        border-radius: 24px;
        padding: 2.8rem 3rem 2.6rem;
        margin-bottom: 2.2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 36px rgba(0, 27, 58, 0.22);
    }

    .hero::after {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 0;
        height: 0;
        border-style: solid;
        border-width: 0 170px 170px 0;
        border-color: transparent var(--fincommerce-aqua) transparent transparent;
        opacity: 0.18;
    }

    .hero-dot {
        position: absolute;
        bottom: -18px;
        right: 60px;
        width: 56px;
        height: 56px;
        background: var(--fincommerce-aqua);
        border-radius: 8px;
        opacity: 0.45;
        transform: rotate(15deg);
    }

    .hero-eyebrow {
        font-family: 'Syne', sans-serif;
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--fincommerce-aqua);
        letter-spacing: 4px;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    .hero-eyebrow::before {
        content: '';
        display: inline-block;
        width: 24px;
        height: 2px;
        background: var(--fincommerce-aqua);
        border-radius: 2px;
    }

    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF;
        margin-bottom: 0.7rem;
        line-height: 1.15;
        letter-spacing: -0.5px;
    }

    .hero-sub {
        font-size: 0.92rem;
        color: rgba(255, 255, 255, 0.72);
        max-width: 580px;
        line-height: 1.7;
        font-weight: 400;
    }

    .hero-pills {
        display: flex;
        gap: 0.75rem;
        margin-top: 1.8rem;
        flex-wrap: wrap;
    }

    .hero-pill {
        background: rgba(17, 214, 190, 0.13);
        border: 1px solid rgba(17, 214, 190, 0.32);
        border-radius: 100px;
        padding: 0.38rem 1.1rem;
        font-size: 0.75rem;
        color: var(--fincommerce-aqua);
        font-weight: 700;
        letter-spacing: 0.3px;
    }

    .hero-pill.orange {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(255, 255, 255, 0.18);
        color: #FFFFFF;
    }

    .section-title {
        font-family: 'Syne', sans-serif;
        font-size: 0.82rem;
        font-weight: 800;
        color: var(--fincommerce-navy);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 2rem 0 1.1rem 0;
        display: flex;
        align-items: center;
        gap: 0.7rem;
    }

    .section-title::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(to right, rgba(0, 53, 102, 0.22), transparent);
    }

    .value-cards {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }

    .value-card {
        background: var(--fincommerce-card);
        border-radius: 16px;
        padding: 1.4rem 1.5rem;
        border-top: 3px solid var(--fincommerce-aqua);
        box-shadow: 0 1px 4px rgba(0, 27, 58, 0.05), 0 4px 16px rgba(0, 27, 58, 0.06);
        transition: box-shadow 0.2s ease, transform 0.2s ease;
    }

    .value-card:hover {
        box-shadow: 0 4px 24px rgba(0, 27, 58, 0.12);
        transform: translateY(-2px);
    }

    .value-card.orange { border-top-color: var(--fincommerce-blue); }
    .value-card.navy { border-top-color: var(--fincommerce-navy); }

    .value-card-icon {
        font-size: 1.6rem;
        margin-bottom: 0.7rem;
        display: block;
    }

    .value-card-title {
        font-family: 'Syne', sans-serif;
        font-size: 0.78rem;
        font-weight: 800;
        color: var(--fincommerce-blue);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.45rem;
    }

    .value-card-text {
        font-size: 0.875rem;
        color: var(--fincommerce-text-muted);
        line-height: 1.6;
        font-weight: 400;
    }

    .stButton > button {
        background: var(--fincommerce-aqua) !important;
        color: var(--fincommerce-navy) !important;
        border: none !important;
        border-radius: 12px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 800 !important;
        font-size: 0.84rem !important;
        padding: 0.65rem 1rem !important;
        transition: all 0.18s ease !important;
        width: 100% !important;
        box-shadow: 0 2px 10px rgba(17, 214, 190, 0.18) !important;
    }

    .stButton > button:hover {
        background: var(--fincommerce-aqua-hover) !important;
        color: var(--fincommerce-navy) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 18px rgba(17, 214, 190, 0.26) !important;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--fincommerce-aqua), var(--fincommerce-aqua-hover)) !important;
        color: var(--fincommerce-navy) !important;
        border: none !important;
        border-radius: 14px !important;
        font-size: 0.95rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.3px !important;
        padding: 0.8rem 1.5rem !important;
        box-shadow: 0 4px 20px rgba(17, 214, 190, 0.28) !important;
    }

    .input-label {
        font-size: 0.83rem;
        font-weight: 800;
        color: var(--fincommerce-navy);
        margin-bottom: 0.15rem;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .input-hint {
        font-size: 0.75rem;
        color: var(--fincommerce-text-muted);
        margin-bottom: 0.55rem;
        line-height: 1.45;
    }

    .stNumberInput input {
        border-radius: 10px !important;
        border: 1.5px solid rgba(0, 53, 102, 0.16) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.9rem !important;
        color: var(--fincommerce-navy) !important;
        background: #FFFFFF !important;
        padding: 0.55rem 0.8rem !important;
        transition: border-color 0.18s ease !important;
    }

    .stNumberInput input:focus {
        border-color: var(--fincommerce-aqua) !important;
        box-shadow: 0 0 0 3px rgba(17, 214, 190, 0.12) !important;
    }

    .result-hero {
        background: linear-gradient(135deg, var(--fincommerce-navy) 0%, var(--fincommerce-blue) 100%);
        border-radius: 20px;
        padding: 2.2rem 2.4rem;
        text-align: center;
        margin-top: 1.8rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 40px rgba(0, 27, 58, 0.22);
    }

    .result-hero::before {
        content: '';
        position: absolute;
        top: -20px;
        right: -20px;
        width: 120px;
        height: 120px;
        background: var(--fincommerce-aqua);
        opacity: 0.12;
        border-radius: 50%;
    }

    .result-hero::after {
        content: '';
        position: absolute;
        bottom: -30px;
        left: 40px;
        width: 100px;
        height: 100px;
        background: var(--fincommerce-aqua-hover);
        opacity: 0.08;
        border-radius: 50%;
    }

    .result-label {
        font-family: 'Syne', sans-serif;
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 4px;
        text-transform: uppercase;
        color: var(--fincommerce-aqua);
        margin-bottom: 0.8rem;
    }

    .result-categoria {
        font-family: 'Syne', sans-serif;
        font-size: 2.1rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.15;
    }

    .confidence-bar-wrap {
        background: rgba(0, 53, 102, 0.10);
        border-radius: 100px;
        height: 7px;
        margin: 0.6rem 0 0.4rem;
        overflow: hidden;
    }

    .confidence-bar {
        height: 100%;
        border-radius: 100px;
        transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
    }

    .biz-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-top: 1rem;
        border-left: 4px solid var(--fincommerce-aqua);
        box-shadow: 0 2px 12px rgba(0, 27, 58, 0.07);
    }

    .biz-title {
        font-family: 'Syne', sans-serif;
        font-size: 0.72rem;
        font-weight: 800;
        color: var(--fincommerce-blue);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.8rem;
    }

    .biz-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }

    .biz-list li {
        font-size: 0.875rem;
        color: var(--fincommerce-text-muted);
        padding: 0.4rem 0;
        border-bottom: 1px solid rgba(0, 53, 102, 0.08);
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
        line-height: 1.55;
    }

    .biz-list li:last-child { border-bottom: none; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--fincommerce-navy) 0%, var(--fincommerce-blue) 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        padding: 0.3rem 0 1.6rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1.4rem;
    }

    .sidebar-logo-badge {
        width: 36px;
        height: 36px;
        background: var(--fincommerce-aqua);
        border-radius: 9px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 0.9rem;
        color: var(--fincommerce-navy) !important;
        flex-shrink: 0;
    }

    .sidebar-logo-name {
        font-family: 'Syne', sans-serif;
        font-size: 1rem;
        font-weight: 800;
        color: #FFFFFF !important;
        line-height: 1.1;
    }

    .sidebar-logo-sub {
        font-size: 0.65rem;
        color: rgba(255,255,255,0.52) !important;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        font-weight: 700;
    }

    .sidebar-section-label {
        font-size: 0.65rem;
        font-weight: 800;
        color: var(--fincommerce-aqua) !important;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
        margin-top: 0.2rem;
    }

    .sidebar-metric {
        background: rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.6rem;
        border-left: 3px solid var(--fincommerce-aqua);
        transition: background 0.15s ease;
    }

    .sidebar-metric:hover { background: rgba(255,255,255,0.09); }
    .sidebar-metric.orange { border-left-color: #FFFFFF; }

    .sidebar-metric-label {
        font-size: 0.68rem;
        color: rgba(255,255,255,0.56) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .sidebar-metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 1.5rem;
        font-weight: 800;
        color: #FFFFFF !important;
        line-height: 1.1;
    }

    .sidebar-info {
        font-size: 0.78rem;
        color: rgba(255,255,255,0.62) !important;
        line-height: 1.65;
    }

    .sidebar-tag {
        display: inline-block;
        background: rgba(17, 214, 190, 0.14);
        border: 1px solid rgba(17, 214, 190, 0.28);
        border-radius: 6px;
        padding: 0.22rem 0.65rem;
        font-size: 0.72rem;
        color: var(--fincommerce-aqua) !important;
        font-weight: 700;
        margin: 0.15rem 0.1rem;
        letter-spacing: 0.3px;
    }

    .sidebar-divider {
        height: 1px;
        background: rgba(255,255,255,0.07);
        margin: 1.2rem 0;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 2px solid rgba(0, 53, 102, 0.12) !important;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        color: var(--fincommerce-text-muted) !important;
    }

    .stTabs [aria-selected="true"] {
        color: var(--fincommerce-navy) !important;
        border-bottom: 2px solid var(--fincommerce-aqua) !important;
    }

    [data-testid="metric-container"] {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 1rem 1.2rem !important;
        border-top: 3px solid var(--fincommerce-aqua);
        box-shadow: 0 2px 10px rgba(0, 27, 58, 0.06);
    }

    [data-testid="metric-container"] label {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        color: var(--fincommerce-text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-family: 'Syne', sans-serif !important;
        font-size: 1.55rem !important;
        font-weight: 800 !important;
        color: var(--fincommerce-navy) !important;
    }

    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--fincommerce-border);
    }

    .stAlert {
        border-radius: 10px;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Datos de negocio por categoría ──────────────────────────────────────────
CATEGORIAS = {
    "hogar_y_decoracion": {
        "emoji": "🏠",
        "nombre": "Hogar y Decoración",
        "color": "#11D6BE",
        "acciones": [
            "Mostrar productos de decoración, muebles y artículos para el hogar",
            "Activar campañas de cross-selling con productos complementarios",
            "Recomendar combos por temporada (mudanzas, renovaciones)",
        ],
        "insight": "Clientes con alto gasto y compras recurrentes tienden a renovar el hogar por etapas. Alta oportunidad de ticket promedio elevado.",
    },
    "electronica_y_tecnologia": {
        "emoji": "💻",
        "nombre": "Electrónica y Tecnología",
        "color": "#001B3A",
        "acciones": [
            "Priorizar accesorios y periféricos complementarios al producto principal",
            "Activar alertas de precio y disponibilidad en categorías de interés",
            "Ofrecer garantías extendidas y servicios de instalación",
        ],
        "insight": "Segmento de alto valor unitario. El cliente que compra electrónica raramente lo hace por impulso, tiene intención clara de compra.",
    },
    "moda_y_belleza": {
        "emoji": "👗",
        "nombre": "Moda y Belleza",
        "color": "#003566",
        "acciones": [
            "Activar recomendaciones basadas en tendencias y temporada",
            "Personalizar catálogo por historial de estilos previos",
            "Ofrecer descuentos por volumen o suscripción a productos recurrentes",
        ],
        "insight": "Alta frecuencia de recompra. Los clientes de moda responden bien a campañas de retención y programas de fidelización.",
    },
    "deporte_ocio_y_juguetes": {
        "emoji": "⚽",
        "nombre": "Deporte, Ocio y Juguetes",
        "color": "#11D6BE",
        "acciones": [
            "Recomendar equipamiento complementario según deporte de interés",
            "Activar campañas estacionales (vacaciones, vuelta al cole)",
            "Ofrecer bundles familiares o por edad objetivo",
        ],
        "insight": "Segmento sensible a estacionalidad. La recencia baja en días desde última compra suele anticipar una nueva intención de compra.",
    },
    "automotriz_y_construccion": {
        "emoji": "🔧",
        "nombre": "Automotriz y Construcción",
        "color": "#003566",
        "acciones": [
            "Recomendar repuestos y accesorios compatibles con productos anteriores",
            "Activar recordatorios de mantenimiento preventivo",
            "Priorizar marcas con alta calificación en categoría",
        ],
        "insight": "Categoría de nicho con clientes de alta lealtad. Bajo volumen pero alto valor por transacción. Estrategia de retención más que adquisición.",
    },
    "otros_y_servicios": {
        "emoji": "📦",
        "nombre": "Otros y Servicios",
        "color": "#6B778C",
        "acciones": [
            "Explorar historial de navegación para identificar intereses específicos",
            "Aplicar encuesta rápida de preferencias para mejorar la predicción",
            "Ofrecer productos de mayor rotación como punto de entrada",
        ],
        "insight": "Perfil difuso. Puede indicar un cliente nuevo o de comportamiento atípico. Recomendación: complementar con datos de sesión para mayor precisión.",
    },
}

# ── Carga de artefactos ──────────────────────────────────────────────────────
@st.cache_resource
def cargar_artefactos():
    """
    Primero intenta cargar el artefacto único:
        models/lightgbm_recommender.joblib

    Si no existe, intenta cargar los tres artefactos separados:
        deployment/lightgbm_model.pkl
        deployment/label_encoder.pkl
        deployment/feature_columns.pkl
    """
    base = Path(__file__).resolve().parent

    single_artifact_path = base / "models" / "lightgbm_recommender.joblib"
    if single_artifact_path.exists():
        artifact = joblib.load(single_artifact_path)
        modelo = artifact["model"]
        encoder = artifact["label_encoder"]
        features = artifact["features"]
        return modelo, encoder, features

    deployment_path = base / "deployment"
    modelo_path = deployment_path / "lightgbm_model.pkl"
    encoder_path = deployment_path / "label_encoder.pkl"
    features_path = deployment_path / "feature_columns.pkl"

    missing_files = [
        path for path in [modelo_path, encoder_path, features_path]
        if not path.exists()
    ]

    if missing_files:
        st.error("No se encontraron los artefactos del modelo.")
        st.write("Primero intenté cargar el artefacto único desde:")
        st.code(str(single_artifact_path), language="text")
        st.write("Luego intenté cargar los artefactos separados desde:")
        st.code(str(deployment_path), language="text")
        st.write("Archivos faltantes:")
        for path in missing_files:
            st.code(str(path), language="text")
        st.stop()

    modelo = joblib.load(modelo_path)
    encoder = joblib.load(encoder_path)
    features = joblib.load(features_path)

    return modelo, encoder, features


modelo, encoder, features = cargar_artefactos()

# ── Feature engineering ──────────────────────────────────────────────────────
def preparar_features(compras, popularidad, rating, gasto, dias):
    d = {
        "customer_purchase_count": compras,
        "product_popularity": popularidad,
        "product_rating": rating,
        "customer_total_spend": gasto,
        "days_since_last_purchase": dias,
    }
    d["log_customer_total_spend"] = np.log1p(gasto)
    d["log_product_popularity"] = np.log1p(popularidad)
    d["log_purchase_count"] = np.log1p(compras)
    d["log_days_since"] = np.log1p(dias)
    d["spend_per_purchase"] = gasto / (compras + 1)
    d["recency_score"] = 1.0 / (dias + 1)
    d["popularity_x_rating"] = popularidad * rating
    d["spend_x_popularity"] = gasto * popularidad
    d["rating_centered"] = rating - 4.07
    d["spend_quartile"] = 0 if gasto < 100 else (1 if gasto < 500 else (2 if gasto < 1500 else 3))
    d["popularity_decile"] = min(int(popularidad / 30), 9)

    input_df = pd.DataFrame([d])

    missing_features = [feature for feature in features if feature not in input_df.columns]
    if missing_features:
        st.error("Faltan features requeridas por el modelo:")
        st.write(missing_features)
        st.stop()

    return input_df[features]

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class='sidebar-logo'>
        <div class='sidebar-logo-badge'>fc</div>
        <div>
            <div class='sidebar-logo-name'>FinCommerce</div>
            <div class='sidebar-logo-sub'>Predictor de Categorías</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">Rendimiento del modelo</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class='sidebar-metric'>
        <div class='sidebar-metric-label'>Accuracy</div>
        <div class='sidebar-metric-value'>70.1%</div>
    </div>
    <div class='sidebar-metric orange'>
        <div class='sidebar-metric-label'>F1 Macro</div>
        <div class='sidebar-metric-value'>69.5%</div>
    </div>
    <div class='sidebar-metric'>
        <div class='sidebar-metric-label'>F1 Weighted</div>
        <div class='sidebar-metric-value'>70.1%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='sidebar-divider'></div>
    <div class='sidebar-section-label'>Stack técnico</div>
    <div class='sidebar-info'>
        <span class='sidebar-tag'>LightGBM</span>
        <span class='sidebar-tag'>scikit-learn</span>
        <span class='sidebar-tag'>MLflow</span>
        <span class='sidebar-tag'>joblib</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='sidebar-divider'></div>
    <div class='sidebar-section-label'>Dataset</div>
    <div class='sidebar-info'>
        Olist · Brazilian E-Commerce<br>
        <strong style='color:white;'>115.694</strong> registros totales<br>
        <strong style='color:white;'>6</strong> macro-categorías<br>
        <strong style='color:white;'>16</strong> features de entrada
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='sidebar-divider'></div>
    <div style='padding:1rem;background:rgba(245,166,35,0.10);border-radius:12px;border:1px solid rgba(245,166,35,0.22);'>
        <div class='sidebar-section-label' style='color:#f5a623 !important;margin-bottom:0.5rem;'>¿Qué hace este modelo?</div>
        <div class='sidebar-info' style='font-size:0.8rem;'>
            Predice la macro-categoría de compra más probable para un cliente dado su comportamiento histórico y el producto evaluado. Actúa como <strong style='color:white;'>capa estratégica</strong> del sistema de recomendación, filtrando el espacio de productos antes del algoritmo colaborativo.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='hero'>
    <div class='hero-dot'></div>
    <div class='hero-eyebrow'>FinCommerce Analytics · Sistema de Recomendación</div>
    <div class='hero-title'>Predictor de<br>Macro-Categorías</div>
    <div class='hero-sub'>
        Clasificamos el perfil de compra de un cliente en una de 6 macro-categorías 
        usando LightGBM, permitiendo personalizar las recomendaciones antes de aplicar 
        el filtrado colaborativo. Ingresá los datos del cliente para obtener la predicción.
    </div>
    <div class='hero-pills'>
        <div class='hero-pill'>📊 70% Accuracy</div>
        <div class='hero-pill'>⚡ Inferencia en tiempo real</div>
        <div class='hero-pill orange'>🎯 6 categorías de negocio</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Valor de negocio ─────────────────────────────────────────────────────────
st.markdown('<div class="section-title">¿Qué valor genera este sistema?</div>', unsafe_allow_html=True)

st.markdown("""
<div class='value-cards'>
    <div class='value-card'>
        <span class='value-card-icon'>🎯</span>
        <div class='value-card-title'>Personalización</div>
        <div class='value-card-text'>Reduce el espacio de búsqueda de productos a la categoría más relevante para cada cliente, mejorando la experiencia de compra.</div>
    </div>
    <div class='value-card orange'>
        <span class='value-card-icon'>🔁</span>
        <div class='value-card-title'>Cold Start</div>
        <div class='value-card-text'>Asigna una categoría probable incluso a clientes con pocas compras, solucionando el problema de datos insuficientes.</div>
    </div>
    <div class='value-card navy'>
        <span class='value-card-icon'>📈</span>
        <div class='value-card-title'>Conversión</div>
        <div class='value-card-text'>Priorizar la categoría correcta antes de recomendar productos aumenta la probabilidad de compra y el ticket promedio.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Perfiles demo ────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">⚡ Perfiles demo — hacé clic para cargar</div>', unsafe_allow_html=True)

if "perfil" not in st.session_state:
    st.session_state.perfil = None

col1, col2, col3, col4 = st.columns(4)

perfiles = {
    "🔄 Cliente frecuente": (10, 200, 4.5, 800.0, 10),
    "🆕 Cliente nuevo": (1, 30, 3.0, 50.0, 120),
    "💤 Cliente inactivo": (2, 80, 4.0, 150.0, 200),
    "⭐ Cliente premium": (15, 350, 4.8, 2000.0, 5),
}

for col, (label, vals) in zip([col1, col2, col3, col4], perfiles.items()):
    with col:
        if st.button(label):
            st.session_state.perfil = vals

defaults = st.session_state.perfil if st.session_state.perfil else (3, 50, 4.5, 500.0, 15)
compras_def, pop_def, rating_def, gasto_def, dias_def = defaults

# ── Inputs ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🔢 Datos del cliente y del producto</div>', unsafe_allow_html=True)

col_a, col_b = st.columns(2)

with col_a:
    st.markdown('<div class="input-label">Cantidad de compras históricas</div><div class="input-hint">Número total de pedidos registrados por el cliente. Mayor frecuencia indica mayor lealtad.</div>', unsafe_allow_html=True)
    compras = st.number_input("compras", min_value=1, value=int(compras_def), label_visibility="collapsed")

    st.markdown('<div class="input-label">Popularidad del producto</div><div class="input-hint">Cantidad de interacciones totales del producto en el catálogo (ventas + visualizaciones).</div>', unsafe_allow_html=True)
    popularidad = st.number_input("popularidad", min_value=1, value=int(pop_def), label_visibility="collapsed")

    st.markdown('<div class="input-label">Rating del producto (1.0 – 5.0)</div><div class="input-hint">Calificación promedio recibida. Incide en la interacción popularidad × rating del modelo.</div>', unsafe_allow_html=True)
    rating = st.number_input("rating", min_value=1.0, max_value=5.0, value=float(rating_def), step=0.1, label_visibility="collapsed")

with col_b:
    st.markdown('<div class="input-label">Gasto total del cliente (R$)</div><div class="input-hint">Suma acumulada de todos los pedidos en la plataforma. Determina el cuartil de gasto.</div>', unsafe_allow_html=True)
    gasto = st.number_input("gasto", min_value=1.0, value=float(gasto_def), label_visibility="collapsed")

    st.markdown('<div class="input-label">Días desde la última compra</div><div class="input-hint">Recencia del cliente. Valores bajos indican mayor probabilidad de recompra inmediata.</div>', unsafe_allow_html=True)
    dias = st.number_input("dias", min_value=0, value=int(dias_def), label_visibility="collapsed")

    st.markdown("""
    <div style='background:#F5F7FA;border:1.5px solid rgba(17,214,190,0.25);border-radius:12px;padding:1rem 1.1rem;margin-top:0.4rem;'>
        <div style='font-size:0.72rem;font-weight:700;color:#003566;margin-bottom:0.3rem;letter-spacing:0.5px;text-transform:uppercase;'>ℹ️ Sobre las features</div>
        <div style='font-size:0.8rem;color:#2c4260;line-height:1.55;'>
            El modelo aplica internamente 11 transformaciones adicionales 
            (log-transforms, ratios, interacciones) antes de predecir. 
            Solo necesitás ingresar las 5 variables base.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Predicción ───────────────────────────────────────────────────────────────
st.markdown("")
predict_btn = st.button("🔍 Predecir categoría de compra", type="primary", use_container_width=True)

if predict_btn:
    input_df = preparar_features(compras, popularidad, rating, gasto, dias)
    pred_num = modelo.predict(input_df)[0]
    pred_label = encoder.inverse_transform([pred_num])[0]
    proba = modelo.predict_proba(input_df)[0]

    cat = CATEGORIAS.get(pred_label, CATEGORIAS["otros_y_servicios"])
    emoji = cat["emoji"]
    nombre = cat["nombre"]
    color = cat["color"]
    # Resultado principal
    st.markdown(f"""
    <div class='result-hero'>
        <div class='result-label'>▸ Categoría predicha por el modelo</div>
        <div class='result-categoria'>{emoji} {nombre}</div>
    </div>
    """, unsafe_allow_html=True)

    # Implicaciones de negocio
    st.markdown(f"""
    <div class='biz-card'>
        <div class='biz-title'>💡 Implicaciones para el negocio</div>
        <ul class='biz-list'>
            {"".join(f"<li>✓ {a}</li>" for a in cat['acciones'])}
        </ul>
        <div style='margin-top:0.9rem;padding:0.75rem 0.9rem;background:#F5F7FA;border-radius:10px;font-size:0.82rem;color:#3a4a5c;line-height:1.55;border-left:3px solid #11D6BE;'>
            <strong style='color:#003566;'>Insight:</strong> {cat['insight']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Probabilidades
    st.markdown('<div class="section-title">📊 Distribución de probabilidades por categoría</div>', unsafe_allow_html=True)

    proba_df = pd.DataFrame({
        "Categoría": [CATEGORIAS.get(c, {"nombre": c})["nombre"] for c in encoder.classes_],
        "Probabilidad (%)": [round(p * 100, 1) for p in proba],
    }).sort_values("Probabilidad (%)", ascending=False)

    st.bar_chart(proba_df.set_index("Categoría"), color="#11D6BE")

    st.markdown(f"""
    <div style='font-size:0.78rem;color:#7a95b0;text-align:center;margin-top:0.6rem;padding:0 1rem;line-height:1.5;'>
        La distribución muestra la probabilidad asignada por LightGBM a cada macro-categoría. 
        Una distribución concentrada indica mayor certeza del modelo.
    </div>
    """, unsafe_allow_html=True)
