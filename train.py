from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from model import DATASET_PATH, FEATURE_COLUMNS, TARGET_COLUMN, train_model

KAGGLE_DATASET_SLUG = "vedauv/what-makes-a-movie-a-blockbuster-5000-films"
MODEL_PATH = Path(__file__).resolve().parent / "modelo.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena el Random Forest para clasificar películas blockbuster."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="CSV o carpeta local. Si se omite, descarga el dataset público de Kaggle.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help=f"Ruta del artefacto de salida. Por defecto: {MODEL_PATH}",
    )
    return parser.parse_args()


def resolve_dataset_path(dataset_root: Path | None) -> Path:
    if dataset_root is None:
        import kagglehub

        print(f"Descargando dataset de Kaggle: {KAGGLE_DATASET_SLUG}")
        dataset_root = Path(kagglehub.dataset_download(KAGGLE_DATASET_SLUG))
    elif not dataset_root.exists():
        raise FileNotFoundError(f"No existe la ruta local indicada: {dataset_root}")
    else:
        print(f"Usando dataset local: {dataset_root.resolve()}")

    if dataset_root.is_file():
        return dataset_root

    required_columns = {*FEATURE_COLUMNS, TARGET_COLUMN}
    for csv_path in sorted(dataset_root.rglob("*.csv")):
        try:
            columns = set(pd.read_csv(csv_path, nrows=0).columns)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            continue
        if required_columns.issubset(columns):
            return csv_path

    raise FileNotFoundError(
        f"No se encontró un CSV con las columnas requeridas dentro de {dataset_root}."
    )


def main() -> None:
    args = parse_args()
    dataset_path = resolve_dataset_path(args.dataset_root)
    model, metrics, genres, _, confusion = train_model(dataset_path)
    training_data = pd.read_csv(dataset_path)
    model.fit(
        training_data[FEATURE_COLUMNS],
        training_data[TARGET_COLUMN].astype(int),
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

    artifact = {
        "model": model,
        "metrics": metrics,
        "genres": genres,
        "feature_importances": feature_importances,
        "confusion": confusion,
        "metadata": {
            "features": FEATURE_COLUMNS,
            "target": TARGET_COLUMN,
            "training_rows": len(training_data),
            "dataset_source": KAGGLE_DATASET_SLUG if args.dataset_root is None else str(dataset_path),
            "sklearn_version": sklearn.__version__,
        },
    }
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.model_path, compress=3)

    print("Métricas del conjunto de prueba:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.3f}")
    print(f"Artefacto guardado en: {args.model_path.resolve()}")


if __name__ == "__main__":
    main()