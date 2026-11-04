from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from model import FEATURE_COLUMNS, TARGET_COLUMN, train_model

KAGGLE_DATASET_SLUG = "vedauv/what-makes-a-movie-a-blockbuster-5000-films"
MODEL_PATH = Path(__file__).resolve().parent / "modelo.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena una red neuronal para clasificar películas blockbuster."
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
    (
        model,
        metrics,
        genres,
        feature_importances,
        confusion,
        experiment_metrics,
    ) = train_model(dataset_path)
    training_data = pd.read_csv(dataset_path)

    artifact = {
        "model": model,
        "metrics": metrics,
        "genres": genres,
        "feature_importances": feature_importances,
        "confusion": confusion,
        "experiment_metrics": experiment_metrics,
        "metadata": {
            "features": FEATURE_COLUMNS,
            "target": TARGET_COLUMN,
            "training_rows": len(training_data),
            "dataset_source": KAGGLE_DATASET_SLUG if args.dataset_root is None else str(dataset_path),
            "model_type": "MLPClassifier",
            "hidden_layer_sizes": (64, 32),
            "class_balancing": "none in baseline; Focal Loss adaptation uses balanced alpha",
            "augmentation": {
                "training_copies": 2,
                "runtime_noise_std_minutes": 2.0,
                "year_noise_std_years": 1.0,
                "preserved_features": ["genre", "budget_usd"],
            },
            "sklearn_version": sklearn.__version__,
        },
    }
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.model_path, compress=3)

    print("Métricas del conjunto de prueba:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.3f}")
    print("\nComparación sobre la misma partición de prueba:")
    print(experiment_metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Artefacto guardado en: {args.model_path.resolve()}")


if __name__ == "__main__":
    main()