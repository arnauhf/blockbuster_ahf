from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.nn import functional as torch_functional
from torch.utils.data import DataLoader, TensorDataset

DATASET_PATH = Path(__file__).resolve().parent / "dataset" / "movie_blockbusters.csv"
TARGET_COLUMN = "blockbuster"
FEATURE_COLUMNS = ["genre", "budget_usd", "runtime_min", "year"]
NUMERIC_FEATURES = ["budget_usd", "runtime_min", "year"]
AUGMENTATION_COPIES = 2
RUNTIME_NOISE_STD_MIN = 2.0
YEAR_NOISE_STD_YEARS = 1.0
NOISY_EVALUATION_COPIES = 5


def _jitter_tabular_features(features: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    augmented = features.copy()
    augmented["runtime_min"] = np.rint(
        augmented["runtime_min"]
        + rng.normal(0, RUNTIME_NOISE_STD_MIN, len(augmented))
    )
    augmented["year"] = np.rint(
        augmented["year"] + rng.normal(0, YEAR_NOISE_STD_YEARS, len(augmented))
    )
    return augmented


def _dense_float32(features: object) -> np.ndarray:
    if hasattr(features, "toarray"):
        features = features.toarray()
    return np.asarray(features, dtype=np.float32)


def _train_adapted_mlp(
    features_train: np.ndarray,
    target_train: pd.Series,
    features_test: np.ndarray,
    *,
    loss: str,
    dropout: float,
    label_smoothing: float,
    random_state: int,
) -> np.ndarray:
    train_features, validation_features, train_target, validation_target = (
        train_test_split(
            features_train,
            target_train.to_numpy(),
            test_size=0.15,
            stratify=target_train,
            random_state=random_state,
        )
    )
    torch.manual_seed(random_state)
    class_counts = np.bincount(train_target)
    class_weights = torch.tensor(
        len(train_target) / (len(class_counts) * class_counts),
        dtype=torch.float32,
    )

    model = nn.Sequential(
        nn.Linear(features_train.shape[1], 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(32, 2),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_features),
            torch.from_numpy(train_target.astype(np.int64)),
        ),
        batch_size=128,
        shuffle=True,
        generator=torch.Generator().manual_seed(random_state),
    )
    validation_x = torch.from_numpy(validation_features)
    validation_y = torch.from_numpy(validation_target.astype(np.int64))
    best_f1 = -1.0
    best_state = deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for _ in range(60):
        model.train()
        for batch_features, batch_target in train_loader:
            optimizer.zero_grad()
            logits = model(batch_features)
            if loss == "focal":
                cross_entropy = torch_functional.cross_entropy(
                    logits, batch_target, reduction="none"
                )
                focal_factor = (1 - torch.exp(-cross_entropy)).pow(2)
                batch_loss = (
                    class_weights[batch_target] * focal_factor * cross_entropy
                ).mean()
            else:
                batch_loss = torch_functional.cross_entropy(
                    logits,
                    batch_target,
                    weight=class_weights,
                    label_smoothing=label_smoothing,
                )
            batch_loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_predictions = model(validation_x).argmax(dim=1).numpy()
        validation_f1 = f1_score(
            validation_y.numpy(),
            validation_predictions,
            average="macro",
            zero_division=0,
        )
        if validation_f1 > best_f1:
            best_f1 = validation_f1
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 8:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(features_test)).argmax(dim=1).numpy()


def train_model(
    dataset_path: str | Path = DATASET_PATH,
) -> tuple[
    Pipeline,
    dict[str, float],
    list[str],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Compare baseline, imbalance, regularization, and tabular augmentation models."""
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
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", transformer),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    solver="adam",
                    alpha=1e-2,
                    learning_rate_init=1e-3,
                    max_iter=500,
                    early_stopping=True,
                    n_iter_no_change=20,
                    random_state=42,
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
    baseline_predictions = model.predict(features_test)

    augmented_features = pd.concat(
        [
            features_train,
            *(
                _jitter_tabular_features(features_train, seed=500 + copy_index)
                for copy_index in range(AUGMENTATION_COPIES)
            ),
        ],
        ignore_index=True,
    )
    augmented_target = pd.concat(
        [target_train] * (AUGMENTATION_COPIES + 1),
        ignore_index=True,
    )
    augmented_model = clone(model)
    augmented_model.fit(augmented_features, augmented_target)
    augmented_predictions = augmented_model.predict(features_test)

    noisy_test_features = pd.concat(
        [
            _jitter_tabular_features(features_test, seed=600 + copy_index)
            for copy_index in range(NOISY_EVALUATION_COPIES)
        ],
        ignore_index=True,
    )
    noisy_test_target = pd.concat(
        [target_test] * NOISY_EVALUATION_COPIES,
        ignore_index=True,
    )
    baseline_noisy_predictions = model.predict(noisy_test_features)
    augmented_noisy_predictions = augmented_model.predict(noisy_test_features)

    positive_class_index = list(augmented_model.classes_).index(1)
    positive_probabilities = augmented_model.predict_proba(features_test)[
        :, positive_class_index
    ]

    preprocessor = model.named_steps["preprocessor"]
    train_matrix = _dense_float32(preprocessor.transform(features_train))
    test_matrix = _dense_float32(preprocessor.transform(features_test))
    noisy_test_matrix = _dense_float32(
        preprocessor.transform(noisy_test_features)
    )
    focal_predictions = _train_adapted_mlp(
        train_matrix,
        target_train,
        test_matrix,
        loss="focal",
        dropout=0.0,
        label_smoothing=0.0,
        random_state=42,
    )
    focal_noisy_predictions = _train_adapted_mlp(
        train_matrix,
        target_train,
        noisy_test_matrix,
        loss="focal",
        dropout=0.0,
        label_smoothing=0.0,
        random_state=42,
    )
    regularized_predictions = _train_adapted_mlp(
        train_matrix,
        target_train,
        test_matrix,
        loss="cross_entropy",
        dropout=0.35,
        label_smoothing=0.1,
        random_state=42,
    )
    regularized_noisy_predictions = _train_adapted_mlp(
        train_matrix,
        target_train,
        noisy_test_matrix,
        loss="cross_entropy",
        dropout=0.35,
        label_smoothing=0.1,
        random_state=42,
    )

    experiment_metrics = pd.DataFrame(
        [
            {
                "modelo": "MLP base (Cross-Entropy)",
                "accuracy": accuracy_score(target_test, baseline_predictions),
                "macro_f1": f1_score(
                    target_test,
                    baseline_predictions,
                    average="macro",
                    zero_division=0,
                ),
                "balanced_accuracy": balanced_accuracy_score(
                    target_test, baseline_predictions
                ),
                "macro_f1_perturbed": f1_score(
                    noisy_test_target,
                    baseline_noisy_predictions,
                    average="macro",
                    zero_division=0,
                ),
            },
            {
                "modelo": "MLP + Focal Loss (gamma=2) + AdamW",
                "accuracy": accuracy_score(target_test, focal_predictions),
                "macro_f1": f1_score(
                    target_test,
                    focal_predictions,
                    average="macro",
                    zero_division=0,
                ),
                "balanced_accuracy": balanced_accuracy_score(
                    target_test, focal_predictions
                ),
                "macro_f1_perturbed": f1_score(
                    noisy_test_target,
                    focal_noisy_predictions,
                    average="macro",
                    zero_division=0,
                ),
            },
            {
                "modelo": "MLP + Dropout (0.35) + Label Smoothing (0.1)",
                "accuracy": accuracy_score(target_test, regularized_predictions),
                "macro_f1": f1_score(
                    target_test,
                    regularized_predictions,
                    average="macro",
                    zero_division=0,
                ),
                "balanced_accuracy": balanced_accuracy_score(
                    target_test, regularized_predictions
                ),
                "macro_f1_perturbed": f1_score(
                    noisy_test_target,
                    regularized_noisy_predictions,
                    average="macro",
                    zero_division=0,
                ),
            },
            {
                "modelo": "MLP + augmentation (runtime ±2 min, año ±1)",
                "accuracy": accuracy_score(target_test, augmented_predictions),
                "macro_f1": f1_score(
                    target_test,
                    augmented_predictions,
                    average="macro",
                    zero_division=0,
                ),
                "balanced_accuracy": balanced_accuracy_score(
                    target_test, augmented_predictions
                ),
                "macro_f1_perturbed": f1_score(
                    noisy_test_target,
                    augmented_noisy_predictions,
                    average="macro",
                    zero_division=0,
                ),
            },
        ]
    )

    metrics = {
        "accuracy": float(accuracy_score(target_test, augmented_predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(target_test, augmented_predictions)
        ),
        "macro_f1": float(
            f1_score(target_test, augmented_predictions, average="macro")
        ),
        "macro_f1_perturbed": float(
            f1_score(
                noisy_test_target,
                augmented_noisy_predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "precision": float(
            precision_score(target_test, augmented_predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(target_test, augmented_predictions, zero_division=0)
        ),
        "f1": float(f1_score(target_test, augmented_predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target_test, positive_probabilities)),
    }
    confusion = pd.DataFrame(
        confusion_matrix(target_test, augmented_predictions, labels=[0, 1]),
        index=["Real: no blockbuster", "Real: blockbuster"],
        columns=["Predicción: no", "Predicción: sí"],
    )

    importance = permutation_importance(
        augmented_model,
        features_test,
        target_test,
        scoring="f1_macro",
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )
    feature_importances = (
        pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance": importance.importances_mean,
            }
        )
        .sort_values("importance", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    augmented_full_features = pd.concat(
        [
            features,
            *(
                _jitter_tabular_features(features, seed=700 + copy_index)
                for copy_index in range(AUGMENTATION_COPIES)
            ),
        ],
        ignore_index=True,
    )
    augmented_full_target = pd.concat(
        [target] * (AUGMENTATION_COPIES + 1),
        ignore_index=True,
    )
    augmented_model.fit(augmented_full_features, augmented_full_target)
    return (
        augmented_model,
        metrics,
        genres,
        feature_importances,
        confusion,
        experiment_metrics,
    )