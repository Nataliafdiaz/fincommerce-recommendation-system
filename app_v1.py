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

st.markdown("""
<style>

/* Fondo general */
.stApp {
    background-color: #F5F7FA;
    color: #001B3A;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #001B3A 0%, #003566 100%);
    color: white;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Título principal */
h1 {
    color: #001B3A;
    font-weight: 800;
}

/* Subtítulos */
h2, h3 {
    color: #001B3A;
    font-weight: 700;
}

/* KPI cards */
div[data-testid="metric-container"] {
    background-color: white;
    border-left: 6px solid #11D6BE;
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

/* Títulos KPI */
[data-testid="stMetricLabel"] {
    color: #4B5563 !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}

[data-testid="stMetricLabel"] > div {
    color: #4B5563 !important;
}

/* Números KPI */
div[data-testid="stMetricValue"] {
    color: #001B3A !important;
    font-size: 34px;
    font-weight: 800;
}

/* Botones */
.stButton button {
    background-color: #11D6BE;
    color: #001B3A;
    border-radius: 12px;
    border: none;
    font-weight: 700;
    padding: 10px;
}

.stButton button:hover {
    background-color: #00F0C8;
    color: #001B3A;
}

/* Selectbox */
div[data-baseweb="select"] {
    background-color: white;
    border-radius: 10px;
}

/* Expander */
.streamlit-expanderHeader {
    color: #001B3A !important;
    font-weight: 700;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}

/* Slider */
.stSlider > div > div {
    color: #11D6BE !important;
}

</style>
""", unsafe_allow_html=True)

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
    if candidates.empty:
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
            ["recommendation_score", "category_probability", "product_rating", "product_popularity"],
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


model_artifact = load_model_artifact(MODEL_PATH)
category_mapping_items = tuple(model_artifact["category_mapping"].items())
df = load_data(DATA_PATH, category_mapping_items)
catalog = build_product_catalog(df)

st.title("Dashboard de Recomendaciones Inteligentes")
st.caption("Sistema de recomendación personalizado para optimizar la experiencia de compra")

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

st.subheader("Métricas generales")

total_spend = df["customer_total_spend"].max()

summary_cols = st.columns(5)

summary_cols[0].metric(
    "Nº de registros",
    f"{len(df):,}"
)

summary_cols[1].metric(
    "Nº de productos",
    f"{df['product_id'].nunique():,}"
)

summary_cols[2].metric(
    "Nº de clientes",
    f"{df['customer_unique_id'].nunique():,}"
)

summary_cols[3].metric(
    "Nº de categorías",
    f"{df['product_category_name'].nunique():,}"
)

summary_cols[4].metric(
    "Gasto total",
    f"${total_spend:,.0f}"
)

if profile["summary"]:
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

recommendations = recommend_products(
    catalog=catalog,
    profile=profile,
    model_artifact=model_artifact,
    top_n=top_n,
    exclude_purchased=exclude_purchased,
)

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

with st.expander("Como se calculan las recomendaciones"):
    st.write(
        "La demo carga el modelo ganador del notebook 3: LightGBM. "
        "El modelo predice la macro-categoria mas probable para el cliente seleccionado. "
        "Luego la app toma productos reales del catalogo solo dentro de esa macro-categoria y los ordena "
        "por probabilidad de categoria, popularidad, rating y afinidad historica del cliente. "
        "El modelo no predice un product_id directamente; el producto se elige en este segundo paso de ranking."
    )

    
    st.markdown("""
    ### Metodología del sistema de recomendación

    El sistema utiliza un modelo de recomendación basado en comportamiento histórico de compra de los usuarios.

    Para generar recomendaciones se consideran variables como:

    - Categorías previamente compradas
    - Popularidad de productos
    - Rating promedio
    - Cantidad de compradores
    - Afinidad entre categorías

    ### Métricas utilizadas

    - Precision@K
    - Recall@K
    - NDCG

    Estas métricas permiten evaluar la relevancia y calidad del ranking de recomendaciones.

    ### Valor de negocio

    El sistema busca:
    - aumentar conversión,
    - mejorar experiencia del usuario,
    - fomentar cross-selling,
    - incrementar retención de clientes.
    """)