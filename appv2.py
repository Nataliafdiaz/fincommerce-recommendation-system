from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


DATA_PATH = Path(__file__).parent / "data" / "dataset_modelo.csv"
MODEL_PATH = Path(__file__).parent / "models" / "lightgbm_recommender.joblib"
REQUIRED_COLUMNS = {
    "customer_unique_id",
    "product_id",
    "product_category_name",
    "customer_purchase_count",
    "product_popularity",
    "product_rating",
    "customer_total_spend",
    "days_since_last_purchase",
}


st.set_page_config(
    page_title="FinCommerce Recommender",
    layout="wide",
)


st.markdown(
    """
    <style>
    :root {
        --fincommerce-navy: #001B3A;
        --fincommerce-blue: #003566;
        --fincommerce-aqua: #11D6BE;
        --fincommerce-aqua-hover: #00F0C8;
        --fincommerce-bg: #F5F7FA;
        --fincommerce-text-muted: #6B778C;
        --fincommerce-dark-panel: #0D1117;
    }

    .stApp {
        background-color: var(--fincommerce-bg);
        color: var(--fincommerce-navy);
    }

    header[data-testid="stHeader"] {
        background-color: var(--fincommerce-dark-panel);
    }

    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--fincommerce-navy) 0%, var(--fincommerce-blue) 100%);
        color: #FFFFFF;
    }

    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    h1 {
        color: var(--fincommerce-navy);
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: 0;
        margin-bottom: 0.25rem;
    }

    h2, h3 {
        color: var(--fincommerce-navy);
        font-weight: 700;
        letter-spacing: 0;
    }

    p, li, .stCaptionContainer {
        color: var(--fincommerce-text-muted);
    }

    div[data-testid="metric-container"] {
        background-color: transparent;
        border: 0;
        border-left: 0;
        padding: 0;
        border-radius: 0;
        box-shadow: none;
    }

    [data-testid="stMetricLabel"] {
        color: var(--fincommerce-text-muted) !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }

    [data-testid="stMetricLabel"] > div {
        color: var(--fincommerce-text-muted) !important;
    }

    div[data-testid="stMetricValue"] {
        color: var(--fincommerce-navy) !important;
        font-size: 2.35rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .stButton button {
        background-color: var(--fincommerce-aqua);
        color: var(--fincommerce-navy) !important;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        padding: 0.65rem 1rem;
        transition: background-color 120ms ease, transform 120ms ease;
    }

    .stButton button:hover {
        background-color: var(--fincommerce-aqua-hover);
        color: var(--fincommerce-navy) !important;
        border: none;
        transform: translateY(-1px);
    }

    .stButton button:focus {
        box-shadow: 0 0 0 0.2rem rgba(17, 214, 190, 0.25);
    }

    section[data-testid="stSidebar"] .stButton button {
        width: 100%;
    }

    div[data-baseweb="select"] > div {
        background-color: #0D1117;
        border-color: transparent;
        border-radius: 10px;
    }

    .stCheckbox [data-testid="stWidgetLabel"] p,
    .stSlider [data-testid="stWidgetLabel"] p,
    .stSelectbox [data-testid="stWidgetLabel"] p {
        font-weight: 600;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 1.25rem;
    }

    .stTabs [data-baseweb="tab"] {
        color: #FFFFFF;
        padding-left: 0;
        padding-right: 0;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        color: #FF5252 !important;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #FF5252;
    }

    [data-testid="stExpander"] {
        background-color: transparent;
        border: none;
        box-shadow: none;
    }

    .streamlit-expanderHeader {
        color: var(--fincommerce-navy) !important;
        font-weight: 700;
    }

    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(0, 27, 58, 0.08);
    }

    .stAlert {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Cargando datos procesados...")
def load_data(path: Path, category_mapping_items: tuple[tuple[str, str], ...]) -> pd.DataFrame:
    if not path.exists():
        st.error(
            "No se encontro data/dataset_modelo.csv. Ejecuta primero el notebook de ETL "
            "o agrega el dataset procesado en la carpeta data."
        )
        st.stop()

    df = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        st.error(
            "El dataset no tiene las columnas requeridas: "
            + ", ".join(sorted(missing_columns))
        )
        st.stop()

    numeric_columns = [
        "customer_purchase_count",
        "product_popularity",
        "product_rating",
        "customer_total_spend",
        "days_since_last_purchase",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["customer_unique_id", "product_id", "product_category_name"])
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    category_mapping = dict(category_mapping_items)
    df["macro_category"] = df["product_category_name"].map(category_mapping)
    df = df.dropna(subset=["macro_category"]).copy()
    return df


@st.cache_resource(show_spinner="Cargando modelo LightGBM...")
def load_model_artifact(path: Path) -> dict:
    if not path.exists():
        st.error(
            "No se encontro el modelo entrenado. Ejecuta primero el notebook "
            "`notebooks/Modelado.ipynb` completo; la celda final guarda "
            "`models/lightgbm_recommender.joblib`."
        )
        st.stop()
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def build_product_catalog(df: pd.DataFrame) -> pd.DataFrame:
    catalog = (
        df.groupby(["product_id", "product_category_name", "macro_category"], as_index=False)
        .agg(
            product_popularity=("product_popularity", "max"),
            product_rating=("product_rating", "mean"),
            buyers=("customer_unique_id", "nunique"),
        )
        .sort_values(
            ["product_popularity", "product_rating", "buyers"],
            ascending=False,
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    catalog["quality_score"] = normalize(catalog["product_rating"])
    catalog["popularity_score"] = normalize(catalog["product_popularity"])
    catalog["base_score"] = (
        0.55 * catalog["popularity_score"]
        + 0.35 * catalog["quality_score"]
        + 0.10 * normalize(catalog["buyers"])
    )
    return catalog


def normalize(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    min_value = values.min()
    max_value = values.max()
    if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
        return pd.Series(np.ones(len(values)), index=values.index)
    return (values - min_value) / (max_value - min_value)


def qcut_codes(values: pd.Series, q: int, use_rank: bool = False) -> pd.Series:
    source = values.rank(method="first") if use_rank else values
    if source.nunique(dropna=True) < 2:
        return pd.Series(np.zeros(len(source), dtype=int), index=source.index)

    codes = pd.qcut(source, q=q, labels=False, duplicates="drop")
    return codes.fillna(0).astype(int)


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df_fe = df.copy()
    df_fe["log_customer_total_spend"] = np.log1p(df_fe["customer_total_spend"])
    df_fe["log_product_popularity"] = np.log1p(df_fe["product_popularity"])
    df_fe["log_purchase_count"] = np.log1p(df_fe["customer_purchase_count"])
    df_fe["log_days_since"] = np.log1p(df_fe["days_since_last_purchase"])
    df_fe["spend_per_purchase"] = (
        df_fe["customer_total_spend"] / (df_fe["customer_purchase_count"] + 1)
    )
    df_fe["recency_score"] = 1.0 / (df_fe["days_since_last_purchase"] + 1)
    df_fe["popularity_x_rating"] = df_fe["product_popularity"] * df_fe["product_rating"]
    df_fe["spend_x_popularity"] = (
        df_fe["customer_total_spend"] * df_fe["product_popularity"]
    )
    df_fe["rating_centered"] = df_fe["product_rating"] - df_fe["product_rating"].mean()
    df_fe["spend_quartile"] = qcut_codes(df_fe["customer_total_spend"], q=4)
    df_fe["popularity_decile"] = qcut_codes(
        df_fe["product_popularity"],
        q=10,
        use_rank=True,
    )
    return df_fe


def customer_profile(df: pd.DataFrame, customer_id: str) -> dict:
    history = df[df["customer_unique_id"] == customer_id]
    if history.empty:
        return {
            "history": history,
            "categories": pd.Series(dtype=float),
            "purchased_products": set(),
            "summary": {},
        }

    categories = history["product_category_name"].value_counts(normalize=True)
    summary = {
        "Compras": int(history["customer_purchase_count"].max()),
        "Gasto total": float(history["customer_total_spend"].max()),
        "Dias desde ultima compra": int(history["days_since_last_purchase"].min()),
        "Categorias compradas": int(history["product_category_name"].nunique()),
    }
    return {
        "history": history,
        "categories": categories,
        "purchased_products": set(history["product_id"]),
        "summary": summary,
    }


def predict_customer_categories(
    profile: dict,
    catalog: pd.DataFrame,
    model_artifact: dict,
) -> pd.DataFrame:
    candidates = catalog.copy()
    if candidates.empty or not profile["summary"]:
        return pd.DataFrame(columns=["macro_category", "category_probability"])

    customer_values = profile["summary"]
    candidates["customer_purchase_count"] = customer_values["Compras"]
    candidates["customer_total_spend"] = customer_values["Gasto total"]
    candidates["days_since_last_purchase"] = customer_values["Dias desde ultima compra"]

    candidates_fe = add_feature_engineering(candidates.reset_index(drop=True))
    feature_matrix = candidates_fe[model_artifact["features"]]
    probabilities = model_artifact["model"].predict_proba(feature_matrix)
    class_labels = model_artifact["label_encoder"].classes_

    category_scores = pd.DataFrame(probabilities, columns=class_labels).mean()
    return (
        category_scores.rename_axis("macro_category")
        .reset_index(name="category_probability")
        .sort_values("category_probability", ascending=False)
        .reset_index(drop=True)
    )


def recommend_products(
    catalog: pd.DataFrame,
    profile: dict,
    model_artifact: dict,
    top_n: int,
    exclude_purchased: bool,
) -> pd.DataFrame:
    candidates = catalog.copy()

    if exclude_purchased and profile["purchased_products"]:
        candidates = candidates[
            ~candidates["product_id"].isin(profile["purchased_products"])
        ].copy()

    if candidates.empty:
        return candidates

    candidates = candidates.reset_index(drop=True)
    category_predictions = predict_customer_categories(
        profile=profile,
        catalog=catalog,
        model_artifact=model_artifact,
    )
    if category_predictions.empty:
        return candidates.iloc[0:0]

    winning_category = category_predictions.iloc[0]["macro_category"]
    candidates = candidates[candidates["macro_category"] == winning_category].copy()
    if candidates.empty:
        return candidates

    candidates = candidates.reset_index(drop=True)
    category_probability = category_predictions.set_index("macro_category")[
        "category_probability"
    ]
    candidates["category_probability"] = (
        candidates["macro_category"].map(category_probability).fillna(0.0)
    )

    category_affinity = profile["categories"]
    candidates["category_affinity"] = (
        candidates["product_category_name"].map(category_affinity).fillna(0.0)
    )

    candidates["recommendation_score"] = (
        0.70 * candidates["category_probability"]
        + 0.20 * candidates["base_score"]
        + 0.10 * candidates["category_affinity"]
    )

    return (
        candidates.sort_values(
            [
                "recommendation_score",
                "category_probability",
                "product_rating",
                "product_popularity",
            ],
            ascending=False,
            kind="mergesort",
        )
        .head(top_n)
        .reset_index(drop=True)
    )


def format_recommendations(recommendations: pd.DataFrame) -> pd.DataFrame:
    if recommendations.empty:
        return recommendations

    output = recommendations[
        [
            "product_id",
            "product_category_name",
            "product_popularity",
            "product_rating",
            "buyers",
            "macro_category",
        ]
    ].copy()
    output.columns = [
        "Producto",
        "Categoria",
        "Popularidad",
        "Rating",
        "Compradores",
        "Macro categoria",
    ]
    output["Rating"] = output["Rating"].round(2)
    return output


def show_customer_profile(profile: dict) -> None:
    if not profile["summary"]:
        st.warning("No se encontro historial para el cliente seleccionado.")
        return

    st.subheader("Perfil del cliente")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Compras", profile["summary"]["Compras"])
    metric_cols[1].metric("Gasto total", f"${profile['summary']['Gasto total']:,.2f}")
    metric_cols[2].metric(
        "Dias desde ultima compra",
        profile["summary"]["Dias desde ultima compra"],
    )
    metric_cols[3].metric(
        "Categorias compradas",
        profile["summary"]["Categorias compradas"],
    )

    with st.expander("Historial reciente del cliente"):
        history = profile["history"][
            [
                "product_id",
                "product_category_name",
                "product_popularity",
                "product_rating",
            ]
        ].drop_duplicates().head(20)
        st.dataframe(history, use_container_width=True, hide_index=True)


model_artifact = load_model_artifact(MODEL_PATH)
category_mapping_items = tuple(model_artifact["category_mapping"].items())
df = load_data(DATA_PATH, category_mapping_items)
catalog = build_product_catalog(df)

with st.sidebar:
    st.header("Parametros")
    top_n = st.slider("Numero de recomendaciones", min_value=3, max_value=20, value=10)
    exclude_purchased = st.checkbox("Excluir productos ya comprados", value=True)

    customer_options = sorted(df["customer_unique_id"].dropna().unique())
    customer_id = st.selectbox(
        "Cliente existente",
        customer_options,
        index=0,
    )
    refresh_recommendations = st.button(
        "Actualizar recomendaciones",
        type="primary",
        use_container_width=True,
    )

profile = customer_profile(df, customer_id)
recommendations = recommend_products(
    catalog=catalog,
    profile=profile,
    model_artifact=model_artifact,
    top_n=top_n,
    exclude_purchased=exclude_purchased,
)

st.title("Dashboard de Recomendaciones Inteligentes")
st.caption("Sistema de recomendacion personalizado para optimizar la experiencia de compra")

tab_dashboard, tab_modelo, tab_metodologia = st.tabs(
    ["Dashboard", "Metricas del modelo", "Metodologia"]
)

with tab_dashboard:
    st.subheader("Metricas generales")

    total_spend = df["customer_total_spend"].max()
    summary_cols = st.columns(5)
    summary_cols[0].metric("No. de registros", f"{len(df):,}")
    summary_cols[1].metric("No. de productos", f"{df['product_id'].nunique():,}")
    summary_cols[2].metric("No. de clientes", f"{df['customer_unique_id'].nunique():,}")
    summary_cols[3].metric("No. de categorias", f"{df['product_category_name'].nunique():,}")
    summary_cols[4].metric("Gasto total", f"${total_spend:,.0f}")

    show_customer_profile(profile)

    st.subheader("Recomendaciones")
    if refresh_recommendations:
        st.success("Recomendaciones actualizadas para el cliente seleccionado.")

    if recommendations.empty:
        st.warning(
            "No hay recomendaciones para la macro-categoria ganadora con los parametros actuales. "
            "Prueba permitiendo productos ya comprados."
        )
    else:
        st.dataframe(
            format_recommendations(recommendations),
            use_container_width=True,
            hide_index=True,
        )

with tab_modelo:
    st.subheader("Metricas del modelo")
    model_metrics = model_artifact.get("metrics", {})

    if model_metrics:
        metric_cols = st.columns(3)
        metric_cols[0].metric("Modelo", model_artifact.get("model_name", "LightGBM"))
        metric_cols[1].metric("F1 weighted", f"{model_metrics.get('f1_weighted', 0):.3f}")
        metric_cols[2].metric("Accuracy", f"{model_metrics.get('accuracy', 0):.3f}")

        st.write("Metricas disponibles en el artefacto del modelo:")
        metrics_df = pd.DataFrame(
            [{"Metrica": key, "Valor": value} for key, value in model_metrics.items()]
        )
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    else:
        st.warning(
            "El archivo del modelo no contiene metricas guardadas en la clave 'metrics'. "
            "La app sigue funcionando para recomendar productos."
        )

    st.subheader("Informacion del artefacto")
    st.write(f"Modelo usado: **{model_artifact.get('model_name', 'LightGBM')}**")
    st.write(
        "Numero de variables usadas por el modelo: "
        f"**{len(model_artifact.get('features', []))}**"
    )

    with st.expander("Variables del modelo"):
        features = model_artifact.get("features", [])
        if features:
            st.dataframe(
                pd.DataFrame({"feature": features}),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No se encontro la lista de features dentro del artefacto del modelo.")

with tab_metodologia:
    st.subheader("Como se calculan las recomendaciones")
    st.write(
        "La demo carga el modelo ganador del notebook 3: LightGBM. "
        "El modelo predice la macro-categoria mas probable para el cliente seleccionado. "
        "Luego la app toma productos reales del catalogo solo dentro de esa macro-categoria y los ordena "
        "por probabilidad de categoria, popularidad, rating y afinidad historica del cliente. "
        "El modelo no predice un product_id directamente; el producto se elige en este segundo paso de ranking."
    )

    st.markdown(
        """
        ### Metodologia del sistema de recomendacion

        El sistema utiliza un modelo de recomendacion basado en comportamiento historico de compra de los usuarios.

        Para generar recomendaciones se consideran variables como:

        - Categorias previamente compradas
        - Popularidad de productos
        - Rating promedio
        - Cantidad de compradores
        - Afinidad entre categorias

        ### Metricas utilizadas

        - Precision@K
        - Recall@K
        - NDCG

        Estas metricas permiten evaluar la relevancia y calidad del ranking de recomendaciones.

        ### Valor de negocio

        El sistema busca:
        - aumentar conversion,
        - mejorar experiencia del usuario,
        - fomentar cross-selling,
        - incrementar retencion de clientes.
        """
    )
