from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import sklearn
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "modelo.joblib"
EXPECTED_ARTIFACT_KEYS = {
    "model",
    "metrics",
    "genres",
    "feature_importances",
    "confusion",
    "experiment_metrics",
    "metadata",
}


@st.cache_resource
def load_artifact() -> dict:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"No se encontró {MODEL_PATH.name}. Ejecuta `uv run python train.py`."
        )

    artifact = joblib.load(MODEL_PATH)
    missing_keys = EXPECTED_ARTIFACT_KEYS.difference(artifact)
    if missing_keys:
        raise ValueError(f"El artefacto no contiene estas claves: {sorted(missing_keys)}")
    if artifact["metadata"].get("model_type") != "MLPClassifier":
        raise ValueError(
            "El artefacto no contiene la red neuronal esperada. "
            "Ejecuta `uv run python train.py` para regenerarlo."
        )
    if artifact["metadata"].get("sklearn_version") != sklearn.__version__:
        raise ValueError(
            "La versión de scikit-learn no coincide con la usada al entrenar. "
            "Ejecuta `uv sync --frozen` y vuelve a generar modelo.joblib."
        )
    return artifact


st.set_page_config(page_title="Detector de Blockbusters", page_icon="🎬", layout="wide")
st.title("¿Será un blockbuster?")
st.caption(
    "Clasificador educativo con una MLP entrenada también con augmentation "
    "ligera de duración y año. "
    "El conjunto de datos es sintético; "
    "el resultado es una estimación para practicar, no una predicción de taquilla real."
)

try:
    artifact = load_artifact()
except (FileNotFoundError, ValueError, KeyError) as error:
    st.error(f"No se pudo cargar el modelo: {error}")
    st.stop()

model = artifact["model"]
metrics = artifact["metrics"]
genres = artifact["genres"]
feature_importances = artifact["feature_importances"]
confusion = artifact["confusion"]
experiment_metrics = artifact["experiment_metrics"]

st.subheader("Evaluación de la red con augmentation")
metric_columns = st.columns(8)
metric_columns[0].metric("Accuracy", f"{metrics['accuracy']:.1%}")
metric_columns[1].metric("Macro F1", f"{metrics['macro_f1']:.1%}")
metric_columns[2].metric("Macro F1 + ruido", f"{metrics['macro_f1_perturbed']:.1%}")
metric_columns[3].metric("Balanced accuracy", f"{metrics['balanced_accuracy']:.1%}")
metric_columns[4].metric("Precisión blockbuster", f"{metrics['precision']:.1%}")
metric_columns[5].metric("Recall blockbuster", f"{metrics['recall']:.1%}")
metric_columns[6].metric("F1 blockbuster", f"{metrics['f1']:.1%}")
metric_columns[7].metric("ROC AUC", f"{metrics['roc_auc']:.3f}")

st.subheader("Comparación de redes")
st.caption(
    "Misma partición reservada para todas las variantes. Macro F1 con ruido "
    "se calcula sobre cinco copias perturbadas de cada película. "
    "Este dataset es tabular y no contiene sinopsis: Global Max Pooling semántico "
    "no aplica aquí."
)
comparison_display = experiment_metrics.rename(
    columns={
        "modelo": "Red",
        "accuracy": "Accuracy",
        "macro_f1": "Macro F1",
        "macro_f1_perturbed": "Macro F1 con ruido",
        "balanced_accuracy": "Balanced accuracy",
    }
)
st.dataframe(
    comparison_display.style.format(
        {
            "Accuracy": "{:.1%}",
            "Macro F1": "{:.1%}",
            "Macro F1 con ruido": "{:.1%}",
            "Balanced accuracy": "{:.1%}",
        }
    ),
    hide_index=True,
    width="stretch",
)

with st.expander("Matriz de confusión y variables importantes"):
    left, right = st.columns(2)
    with left:
        st.caption("Filas: valor real. Columnas: predicción.")
        st.dataframe(confusion, width="stretch")
    with right:
        chart_data = feature_importances.copy()
        chart_data["feature"] = chart_data["feature"].replace(
            {
                "genre": "Género",
                "budget_usd": "Presupuesto",
                "runtime_min": "Duración",
                "year": "Año",
            }
        )
        st.caption("Importancia estimada por permutación (macro F1).")
        st.bar_chart(chart_data, x="feature", y="importance", horizontal=True)

st.divider()
st.subheader("Probar una película")
st.write("Introduce datos que podrían conocerse antes del estreno.")

with st.form("movie_prediction"):
    col_one, col_two = st.columns(2)
    with col_one:
        genre = st.selectbox("Género", genres)
        budget = st.number_input(
            "Presupuesto (USD)",
            min_value=100_000.0,
            max_value=2_000_000_000.0,
            value=50_000_000.0,
            step=5_000_000.0,
            format="%.0f",
        )
    with col_two:
        runtime = st.slider("Duración (minutos)", min_value=60, max_value=180, value=110)
        year = st.slider("Año de estreno", min_value=1980, max_value=2024, value=2024)

    threshold = st.slider(
        "Umbral para clasificar como blockbuster",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        format="%.2f",
    )
    submitted = st.form_submit_button("Calcular predicción", type="primary")

if submitted:
    features = pd.DataFrame(
        [{"genre": genre, "budget_usd": budget, "runtime_min": runtime, "year": year}]
    )
    positive_class_index = list(model.classes_).index(1)
    probability = float(model.predict_proba(features)[0][positive_class_index])
    is_blockbuster = probability >= threshold

    if is_blockbuster:
        st.success("Predicción: blockbuster")
    else:
        st.info("Predicción: no blockbuster")
    st.metric("Probabilidad estimada de blockbuster", f"{probability:.1%}")
    st.caption(f"Clasificación basada en el umbral seleccionado: {threshold:.0%}.")