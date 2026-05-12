import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="FinCommerce Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paleta FinCommerce ───────────────────────────────────────────────────────
# Azul oscuro: #0f2744 | Verde agua: #00c9b1 | Naranja: #f5a623

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #f7f9fc;
    }

    /* ── Header hero ── */
    .hero {
        background: linear-gradient(135deg, #0f2744 0%, #1a3d6b 100%);
        border-radius: 20px;
        padding: 2.5rem 2.8rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -40px; right: -40px;
        width: 220px; height: 220px;
        background: #00c9b1;
        opacity: 0.12;
        border-radius: 50%;
    }
    .hero::after {
        content: '';
        position: absolute;
        bottom: -60px; right: 120px;
        width: 160px; height: 160px;
        background: #f5a623;
        opacity: 0.10;
        border-radius: 50%;
    }
    .hero-brand {
        font-family: 'Syne', sans-serif;
        font-size: 0.85rem;
        font-weight: 600;
        color: #00c9b1;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.6rem;
        line-height: 1.2;
    }
    .hero-sub {
        font-size: 0.95rem;
        color: #a8bdd4;
        max-width: 640px;
        line-height: 1.6;
    }
    .hero-value {
        display: flex;
        gap: 2rem;
        margin-top: 1.5rem;
    }
    .hero-pill {
        background: rgba(0,201,177,0.15);
        border: 1px solid rgba(0,201,177,0.35);
        border-radius: 50px;
        padding: 0.35rem 1rem;
        font-size: 0.78rem;
        color: #00c9b1;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* ── Sección títulos ── */
    .section-title {
        font-family: 'Syne', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        color: #0f2744;
        border-left: 4px solid #00c9b1;
        padding-left: 0.75rem;
        margin: 1.8rem 0 1rem 0;
    }

    /* ── Tarjetas de valor ── */
    .value-cards {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
    }
    .value-card {
        background: white;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        flex: 1;
        min-width: 180px;
        border-top: 3px solid #00c9b1;
        box-shadow: 0 2px 12px rgba(15,39,68,0.07);
    }
    .value-card.orange { border-top-color: #f5a623; }
    .value-card.blue   { border-top-color: #0f2744; }
    .value-card-icon { font-size: 1.4rem; margin-bottom: 0.4rem; }
    .value-card-title {
        font-family: 'Syne', sans-serif;
        font-size: 0.8rem;
        font-weight: 700;
        color: #6b7c93;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.3rem;
    }
    .value-card-text {
        font-size: 0.88rem;
        color: #1a2e44;
        line-height: 1.5;
    }

    /* ── Perfiles demo ── */
    .stButton > button {
        background: white !important;
        color: #0f2744 !important;
        border: 1.5px solid #d0dcea !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        border-color: #00c9b1 !important;
        color: #00c9b1 !important;
        box-shadow: 0 2px 8px rgba(0,201,177,0.2) !important;
    }

    /* ── Input box ── */
    .input-card {
        background: white;
        border-radius: 16px;
        padding: 1.6rem;
        box-shadow: 0 2px 12px rgba(15,39,68,0.06);
        margin-bottom: 1rem;
    }
    .input-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #0f2744;
        margin-bottom: 0.2rem;
    }
    .input-hint {
        font-size: 0.75rem;
        color: #8a9db5;
        margin-bottom: 0.6rem;
        line-height: 1.4;
    }

    /* ── Resultado ── */
    .result-hero {
        background: linear-gradient(135deg, #0f2744 0%, #1a3d6b 100%);
        border-radius: 18px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .result-hero::before {
        content: '';
        position: absolute;
        top: -30px; left: -30px;
        width: 150px; height: 150px;
        background: #00c9b1;
        opacity: 0.08;
        border-radius: 50%;
    }
    .result-label {
        font-family: 'Syne', sans-serif;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #00c9b1;
        margin-bottom: 0.6rem;
    }
    .result-categoria {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.4rem;
    }

    /* ── Implicaciones de negocio ── */
    .biz-card {
        background: white;
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        margin-top: 1rem;
        border-left: 4px solid #f5a623;
        box-shadow: 0 2px 10px rgba(15,39,68,0.06);
    }
    .biz-title {
        font-family: 'Syne', sans-serif;
        font-size: 0.78rem;
        font-weight: 700;
        color: #f5a623;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.6rem;
    }
    .biz-list {
        list-style: none;
        padding: 0; margin: 0;
    }
    .biz-list li {
        font-size: 0.88rem;
        color: #1a2e44;
        padding: 0.3rem 0;
        border-bottom: 1px solid #f0f4f8;
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
        line-height: 1.5;
    }
    .biz-list li:last-child { border-bottom: none; }

    /* ── Confianza ── */
    .confidence-bar-wrap {
        background: #edf2f7;
        border-radius: 50px;
        height: 8px;
        margin: 0.5rem 0 0.3rem;
        overflow: hidden;
    }
    .confidence-bar {
        height: 100%;
        border-radius: 50px;
        transition: width 0.6s ease;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #0f2744 !important;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    .sidebar-metric {
        background: rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.7rem;
        border-left: 3px solid #00c9b1;
    }
    .sidebar-metric.orange { border-left-color: #f5a623; }
    .sidebar-metric-label {
        font-size: 0.72rem;
        color: #a8bdd4 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    .sidebar-metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 1.5rem;
        font-weight: 800;
        color: white !important;
    }
    .sidebar-info {
        font-size: 0.8rem;
        color: #a8bdd4 !important;
        line-height: 1.6;
    }
    .sidebar-tag {
        display: inline-block;
        background: rgba(0,201,177,0.18);
        border: 1px solid rgba(0,201,177,0.35);
        border-radius: 6px;
        padding: 0.2rem 0.6rem;
        font-size: 0.75rem;
        color: #00c9b1 !important;
        font-weight: 600;
        margin: 0.15rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Datos de negocio por categoría ──────────────────────────────────────────
CATEGORIAS = {
    "hogar_y_decoracion": {
        "emoji": "🏠",
        "nombre": "Hogar y Decoración",
        "color": "#00c9b1",
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
        "color": "#0f2744",
        "acciones": [
            "Priorizar accesorios y periféricos complementarios al producto principal",
            "Activar alertas de precio y disponibilidad en categorías de interés",
            "Ofrecer garantías extendidas y servicios de instalación",
        ],
        "insight": "Segmento de alto valor unitario. El cliente que compra electrónica raramente lo hace por impulso — tiene intención clara de compra.",
    },
    "moda_y_belleza": {
        "emoji": "👗",
        "nombre": "Moda y Belleza",
        "color": "#f5a623",
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
        "color": "#00c9b1",
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
        "color": "#1a3d6b",
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
        "color": "#8a9db5",
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
    base = os.path.dirname(os.path.dirname(__file__))
    modelo  = joblib.load(os.path.join(base, "deployment", "lightgbm_model.pkl"))
    encoder = joblib.load(os.path.join(base, "deployment", "label_encoder.pkl"))
    features = joblib.load(os.path.join(base, "deployment", "feature_columns.pkl"))
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
    d["log_product_popularity"]   = np.log1p(popularidad)
    d["log_purchase_count"]       = np.log1p(compras)
    d["log_days_since"]           = np.log1p(dias)
    d["spend_per_purchase"]       = gasto / (compras + 1)
    d["recency_score"]            = 1.0 / (dias + 1)
    d["popularity_x_rating"]      = popularidad * rating
    d["spend_x_popularity"]       = gasto * popularidad
    d["rating_centered"]          = rating - 4.07
    d["spend_quartile"]           = 0 if gasto < 100 else (1 if gasto < 500 else (2 if gasto < 1500 else 3))
    d["popularity_decile"]        = min(int(popularidad / 30), 9)
    return pd.DataFrame([d])[features]

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding: 0.5rem 0 1.5rem;'>
        <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;color:white;'>
            fc <span style='color:#00c9b1;'>·</span> FinCommerce
        </div>
        <div style='font-size:0.75rem;color:#a8bdd4;margin-top:0.2rem;letter-spacing:1px;'>
            PREDICTOR DE CATEGORÍAS
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-family:Syne,sans-serif;font-size:0.7rem;font-weight:700;color:#00c9b1;letter-spacing:2px;text-transform:uppercase;margin-bottom:0.8rem;">Rendimiento del modelo</div>', unsafe_allow_html=True)

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
    <div style='margin-top:1.5rem;margin-bottom:0.6rem;font-family:Syne,sans-serif;font-size:0.7rem;font-weight:700;color:#00c9b1;letter-spacing:2px;text-transform:uppercase;'>
        Stack técnico
    </div>
    <div class='sidebar-info'>
        <span class='sidebar-tag'>LightGBM</span>
        <span class='sidebar-tag'>scikit-learn</span>
        <span class='sidebar-tag'>MLflow</span>
        <span class='sidebar-tag'>joblib</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:1.5rem;margin-bottom:0.6rem;font-family:Syne,sans-serif;font-size:0.7rem;font-weight:700;color:#00c9b1;letter-spacing:2px;text-transform:uppercase;'>
        Dataset
    </div>
    <div class='sidebar-info'>
        Olist · Brazilian E-Commerce<br>
        <strong style='color:white;'>115.694</strong> registros totales<br>
        <strong style='color:white;'>6</strong> macro-categorías<br>
        <strong style='color:white;'>16</strong> features de entrada
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:1.5rem;padding:1rem;background:rgba(245,166,35,0.12);border-radius:12px;border:1px solid rgba(245,166,35,0.25);'>
        <div style='font-size:0.72rem;font-weight:700;color:#f5a623;letter-spacing:1px;text-transform:uppercase;margin-bottom:0.4rem;'>
            ¿Qué hace este modelo?
        </div>
        <div class='sidebar-info' style='font-size:0.82rem;'>
            Predice la macro-categoría de compra más probable para un cliente dado su comportamiento histórico y el producto evaluado. Actúa como <strong style='color:white;'>capa estratégica</strong> del sistema de recomendación, filtrando el espacio de productos antes del algoritmo colaborativo.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='hero'>
    <div class='hero-brand'>FinCommerce Analytics · Sistema de Recomendación</div>
    <div class='hero-title'>Predictor de Macro-Categorías</div>
    <div class='hero-sub'>
        Clasificamos el perfil de compra de un cliente en una de 6 macro-categorías 
        usando LightGBM, permitiendo personalizar las recomendaciones antes de aplicar 
        el filtrado colaborativo. Ingresá los datos del cliente para obtener la predicción.
    </div>
    <div class='hero-value'>
        <div class='hero-pill'>📊 70% Accuracy</div>
        <div class='hero-pill'>⚡ Inferencia en tiempo real</div>
        <div class='hero-pill'>🎯 6 categorías de negocio</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Valor de negocio ─────────────────────────────────────────────────────────
st.markdown('<div class="section-title">¿Qué valor genera este sistema?</div>', unsafe_allow_html=True)

st.markdown("""
<div class='value-cards'>
    <div class='value-card'>
        <div class='value-card-icon'>🎯</div>
        <div class='value-card-title'>Personalización</div>
        <div class='value-card-text'>Reduce el espacio de búsqueda de productos a la categoría más relevante para cada cliente, mejorando la experiencia de compra.</div>
    </div>
    <div class='value-card orange'>
        <div class='value-card-icon'>🔁</div>
        <div class='value-card-title'>Cold Start</div>
        <div class='value-card-text'>Asigna una categoría probable incluso a clientes con pocas compras, solucionando el problema de datos insuficientes.</div>
    </div>
    <div class='value-card blue'>
        <div class='value-card-icon'>📈</div>
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
    "🆕 Cliente nuevo":     (1,  30,  3.0,  50.0, 120),
    "💤 Cliente inactivo":  (2,  80,  4.0, 150.0, 200),
    "⭐ Cliente premium":   (15, 350, 4.8, 2000.0, 5),
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
    <div style='background:#f0fdf9;border:1px solid #00c9b1;border-radius:12px;padding:1rem;margin-top:0.3rem;'>
        <div style='font-size:0.75rem;font-weight:700;color:#00c9b1;margin-bottom:0.3rem;'>ℹ️ Sobre las features</div>
        <div style='font-size:0.8rem;color:#1a2e44;line-height:1.5;'>
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
    input_df   = preparar_features(compras, popularidad, rating, gasto, dias)
    pred_num   = modelo.predict(input_df)[0]
    pred_label = encoder.inverse_transform([pred_num])[0]
    proba      = modelo.predict_proba(input_df)[0]

    cat   = CATEGORIAS.get(pred_label, CATEGORIAS["otros_y_servicios"])
    emoji = cat["emoji"]
    nombre = cat["nombre"]
    color  = cat["color"]
    confianza = proba.max() * 100

    # Resultado principal
    st.markdown(f"""
    <div class='result-hero'>
        <div class='result-label'>▸ Categoría predicha por el modelo</div>
        <div class='result-categoria'>{emoji} {nombre}</div>
    </div>
    """, unsafe_allow_html=True)

    # Confianza
    if confianza >= 60:
        conf_color = "#00c9b1"
        conf_msg   = "Alta confianza — recomendación confiable"
        conf_icon  = "🟢"
    elif confianza >= 40:
        conf_color = "#f5a623"
        conf_msg   = "Confianza moderada — considerar segunda opción"
        conf_icon  = "🟡"
    else:
        conf_color = "#e05c5c"
        conf_msg   = "Baja confianza — perfil atípico o datos insuficientes"
        conf_icon  = "🔴"

    st.markdown(f"""
    <div style='margin-top:1rem;padding:1rem 1.2rem;background:white;border-radius:12px;box-shadow:0 2px 8px rgba(15,39,68,0.06);'>
        <div style='font-size:0.75rem;font-weight:700;color:#6b7c93;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.5rem;'>
            Confianza del modelo
        </div>
        <div style='font-family:Syne,sans-serif;font-size:1.6rem;font-weight:800;color:{conf_color};'>
            {confianza:.1f}%
        </div>
        <div class='confidence-bar-wrap'>
            <div class='confidence-bar' style='width:{confianza}%;background:{conf_color};'></div>
        </div>
        <div style='font-size:0.82rem;color:#4a5568;'>{conf_icon} {conf_msg}</div>
    </div>
    """, unsafe_allow_html=True)

    # Implicaciones de negocio
    st.markdown(f"""
    <div class='biz-card'>
        <div class='biz-title'>💡 Implicaciones para el negocio</div>
        <ul class='biz-list'>
            {"".join(f"<li>✓ {a}</li>" for a in cat['acciones'])}
        </ul>
        <div style='margin-top:0.8rem;padding:0.7rem 0.9rem;background:#fffbf0;border-radius:8px;font-size:0.82rem;color:#4a5568;line-height:1.5;'>
            <strong>Insight:</strong> {cat['insight']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Probabilidades
    st.markdown('<div class="section-title">📊 Distribución de probabilidades por categoría</div>', unsafe_allow_html=True)

    proba_df = pd.DataFrame({
        "Categoría": [CATEGORIAS.get(c, {"nombre": c})["nombre"] for c in encoder.classes_],
        "Probabilidad (%)": [round(p * 100, 1) for p in proba],
    }).sort_values("Probabilidad (%)", ascending=False)

    st.bar_chart(proba_df.set_index("Categoría"), color="#00c9b1")

    st.markdown(f"""
    <div style='font-size:0.8rem;color:#8a9db5;text-align:center;margin-top:0.5rem;'>
        La distribución muestra la probabilidad asignada por LightGBM a cada macro-categoría. 
        Una distribución concentrada indica mayor certeza del modelo.
    </div>
    """, unsafe_allow_html=True)
