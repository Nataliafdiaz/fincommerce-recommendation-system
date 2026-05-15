# FinCommerce Recommendation System

Proyecto de Machine Learning enfocado en el análisis del dataset brasileño de e-commerce de Olist y en la construcción de un sistema de recomendación de productos para clientes existentes.

La solución integra limpieza y transformación de datos, análisis exploratorio, modelado con algoritmos supervisados y una aplicación interactiva en Streamlit con el estilo visual corporativo de FinCommerce.

---

## Descripción General

Este proyecto permite:

- Limpiar, transformar y preparar datos transaccionales de e-commerce.
- Construir variables para modelado a partir de comportamiento de clientes y productos.
- Analizar categorías, clientes, productos, popularidad y gasto.
- Entrenar y comparar modelos de Machine Learning.
- Guardar el modelo ganador LightGBM como artefacto reutilizable.
- Generar recomendaciones personalizadas desde una app unificada en `app.py`.
- Consultar métricas generales, métricas del modelo y metodología desde una sola interfaz.

---

## Dataset

Fuente: Olist Brazilian E-Commerce Dataset.

El dataset se obtiene durante el flujo de notebooks y se transforma en un archivo procesado para modelado:

```text
data/dataset_modelo.csv
```

La app Streamlit usa este archivo como entrada principal.

## Notebooks

### `01_EDA_Explorativo.ipynb`

- Análisis exploratorio de datos.
- Revisión de clientes, productos, categorías y comportamiento de compra.
- Visualizaciones e insights de negocio.

### `02_EDA_ETL.ipynb`

- Limpieza y transformación de datos.
- Construcción de variables derivadas.
- Preparación del dataset final usado por el modelo y la app.

Output principal:

```text
data/dataset_modelo.csv
```

### `03_Modelado.ipynb`

- Preprocesamiento de variables.
- Entrenamiento y comparación de modelos.
- Evaluación de desempeño.
- Selección del modelo ganador.

Output principal:

```text
models/lightgbm_recommender.joblib
```

### `04_Validacion_despliegue.ipynb`

- Validaciones finales para despliegue.
- Pruebas del artefacto entrenado y del flujo de recomendación.

---

## Tecnologías Utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- LightGBM
- Streamlit
- Joblib
- Jupyter Notebook

---

## Cómo Reproducir el Proyecto

### 1. Clonar el repositorio

```bash
git clone https://github.com/Nataliafdiaz/fincommerce-recommendation-system.git
cd fincommerce-recommendation-system
```

### 2. Crear y activar un entorno virtual

Windows:

```bash
py -m venv .venv
.venv\Scripts\activate
```

Mac/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
py -m pip install -r requirements.txt
```

En Mac/Linux, si no usas `py`, ejecuta:

```bash
python -m pip install -r requirements.txt
```

### 4. Ejecutar los notebooks en orden

```text
1. notebooks/01_EDA_Explorativo.ipynb
2. notebooks/02_EDA_ETL.ipynb
3. notebooks/03_Modelado.ipynb
4. notebooks/04_Validacion_despliegue.ipynb
```

Los archivos necesarios para la app son:

```text
data/dataset_modelo.csv
models/lightgbm_recommender.joblib
```

---

## App Unificada con Streamlit

El proyecto incluye una aplicación interactiva en `app.py` que combina las dos versiones previas en una sola experiencia.

La interfaz mantiene la paleta visual de FinCommerce:

- Azul corporativo principal: `#001B3A`
- Azul secundario: `#003566`
- Turquesa de acción: `#11D6BE`
- Fondo claro: `#F5F7FA`

### Ejecutar la app

```bash
py -m streamlit run app.py
```

En Mac/Linux:

```bash
python -m streamlit run app.py
```

Luego abre la URL local que muestra Streamlit, normalmente:

```text
http://localhost:8501
```

### Secciones de la app

#### Dashboard

- Métricas generales del dataset.
- Número de registros, productos, clientes y categorías.
- Gasto total.
- Perfil del cliente seleccionado.
- Historial reciente del cliente.
- Tabla de recomendaciones personalizadas.

#### Métricas del modelo

- Nombre del modelo usado.
- Accuracy.
- F1 weighted.
- Métricas disponibles guardadas en el artefacto.
- Variables utilizadas por el modelo.

#### Metodología

- Explicación del flujo de recomendación.
- Variables consideradas.
- Métricas de evaluación.
- Valor de negocio del sistema.

---

## Lógica de Recomendación

La app carga el artefacto:

```text
models/lightgbm_recommender.joblib
```

El modelo LightGBM predice la macro-categoría más probable para el cliente seleccionado. Luego, la app toma productos reales del catálogo dentro de esa macro-categoría y los ordena usando:

- Probabilidad predicha para la macro-categoría.
- Popularidad del producto.
- Rating promedio.
- Cantidad de compradores.
- Afinidad histórica del cliente con categorías compradas.

El modelo no predice directamente un `product_id`; el producto final se elige en una segunda etapa de ranking.

---

## Casos Contemplados en la App

- Si falta `data/dataset_modelo.csv`, la app indica que se debe ejecutar primero el flujo de ETL.
- Si falta `models/lightgbm_recommender.joblib`, la app indica que se debe ejecutar completo el notebook de modelado.
- Si faltan columnas requeridas, la app muestra cuáles columnas hacen falta.
- Si no hay recomendaciones con los filtros actuales, la app sugiere permitir productos ya comprados.
- La selección de cliente se limita a clientes existentes en el dataset procesado.

---

## Modelos de Machine Learning

El proyecto compara modelos supervisados para clasificación de macro-categorías:

- Random Forest Classifier
- XGBoost Classifier
- LightGBM Classifier

Métricas evaluadas:

- Accuracy
- Precision
- Recall
- F1-Score

El modelo ganador usado por la app es LightGBM.

---
