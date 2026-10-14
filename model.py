from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATASET_PATH = Path(__file__).resolve().parent / "dataset" / "movie_blockbusters.csv"
TARGET_COLUMN = "blockbuster"
FEATURE_COLUMNS = ["genre", "budget_usd", "runtime_min", "year"]
NUMERIC_FEATURES = ["budget_usd", "runtime_min", "year"]


def train_model(
    dataset_path: str | Path = DATASET_PATH,
) -> tuple[Pipeline, dict[str, float], list[str], pd.DataFrame, pd.DataFrame]:
    """Train and evaluate a Random Forest without post-release or target leakage."""
    dataset_path = Path(dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"No se encontró el dataset: {dataset_path}")

    data = pd.read_csv(dataset_path)
    required_columns = {*FEATURE_COLUMNS, TARGET_COLUMN}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing_columns)}")
    if data.empty:
        raise ValueError("El dataset no contiene películas.")
    if data[list(required_columns)].isna().any().any():
        raise ValueError("Las columnas usadas por el modelo contienen valores vacíos.")

    features = data[FEATURE_COLUMNS].copy()
    target = pd.to_numeric(data[TARGET_COLUMN], errors="raise").astype(int)
    if set(target.unique()) != {0, 1}:
        raise ValueError("La columna 'blockbuster' debe contener las clases 0 y 1.")

    genres = sorted(features["genre"].astype(str).unique().tolist())
    transformer = ColumnTransformer(
        transformers=[
            ("genre", OneHotEncoder(handle_unknown="ignore"), ["genre"]),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", transformer),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced_subsample",
                    max_features="sqrt",
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    features_train, features_test, target_train, target_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )
    model.fit(features_train, target_train)
    predictions = model.predict(features_test)
    positive_class_index = list(model.classes_).index(1)
    positive_probabilities = model.predict_proba(features_test)[:, positive_class_index]

    metrics = {
        "accuracy": float(accuracy_score(target_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(target_test, predictions)),
        "precision": float(precision_score(target_test, predictions, zero_division=0)),
        "recall": float(recall_score(target_test, predictions, zero_division=0)),
        "f1": float(f1_score(target_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target_test, positive_probabilities)),
    }
    confusion = pd.DataFrame(
        confusion_matrix(target_test, predictions, labels=[0, 1]),
        index=["Real: no blockbuster", "Real: blockbuster"],
        columns=["Predicción: no", "Predicción: sí"],
    )

    classifier = model.named_steps["classifier"]
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    feature_importances = (
        pd.DataFrame(
            {"feature": feature_names, "importance": classifier.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    return model, metrics, genres, feature_importances, confusion