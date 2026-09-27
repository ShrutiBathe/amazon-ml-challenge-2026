from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "m3_matcher.joblib"
)


# These columns identify the pair and must NOT be given
# directly to the ML model.
ID_COLUMNS = {
    "source1_id",
    "candidate_id",
    "candidate_source",
}


# ============================================================
# FEATURE SELECTION
# ============================================================

def get_feature_columns(features):
    """
    Return only numeric ML feature columns.

    ID columns, candidate source, and the training label
    are never allowed to become model input features.
    """

    excluded_columns = {
        "source1_id",
        "candidate_id",
        "candidate_source",
        "label",
    }

    feature_columns = []

    for column in features.columns:

        if column in excluded_columns:
            continue

        if pd.api.types.is_numeric_dtype(features[column]):
            feature_columns.append(column)

    if not feature_columns:
        raise ValueError(
            "No numeric ML feature columns were found."
        )

    return feature_columns

# ============================================================
# PREPARE TRAINING DATA
# ============================================================

def prepare_training_data(
    features: pd.DataFrame,
    training_pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Combine M3 features with training labels.

    Expected training_pairs columns:
        source1_id
        candidate_id
        candidate_source
        label
    """

    required_feature_columns = {
        "source1_id",
        "candidate_id",
        "candidate_source",
    }

    required_label_columns = {
        "source1_id",
        "candidate_id",
        "candidate_source",
        "label",
    }

    missing_features = (
        required_feature_columns
        - set(features.columns)
    )

    if missing_features:
        raise ValueError(
            "Missing feature columns: "
            f"{sorted(missing_features)}"
        )

    missing_labels = (
        required_label_columns
        - set(training_pairs.columns)
    )

    if missing_labels:
        raise ValueError(
            "Missing training-pair columns: "
            f"{sorted(missing_labels)}"
        )

    # --------------------------------------------------------
    # Keep one label per candidate pair
    # --------------------------------------------------------

    labels = training_pairs[
        [
            "source1_id",
            "candidate_id",
            "candidate_source",
            "label",
        ]
    ].copy()

    labels = labels.drop_duplicates(
        subset=[
            "source1_id",
            "candidate_id",
            "candidate_source",
        ]
    )

    # --------------------------------------------------------
    # Merge features with labels
    # --------------------------------------------------------

    data = features.merge(
        labels,
        on=[
            "source1_id",
            "candidate_id",
            "candidate_source",
        ],
        how="inner",
    )

    if data.empty:
        raise ValueError(
            "No feature rows matched training labels."
        )

    # --------------------------------------------------------
    # Select numeric features
    # --------------------------------------------------------

    feature_columns = get_feature_columns(
        data
    )

    X = data[
        feature_columns
    ].copy()

    y = data["label"].astype(int)

    # --------------------------------------------------------
    # Clean invalid numeric values
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X = X.fillna(0.0)

    return X, y, feature_columns


# ============================================================
# TRAIN MATCHER
# ============================================================

def train_matcher(
    features: pd.DataFrame,
    training_pairs: pd.DataFrame,
    n_estimators: int = 300,
    random_state: int = 42,
) -> tuple[RandomForestClassifier, list[str]]:
    """
    Train the M3 entity-matching classifier.

    RandomForest is used because:

    - handles nonlinear feature interactions
    - works well with mixed evidence features
    - does not require feature scaling
    - supports class weighting
    """

    X, y, feature_columns = prepare_training_data(
        features,
        training_pairs,
    )

    positive_count = int(
        (y == 1).sum()
    )

    negative_count = int(
        (y == 0).sum()
    )

    print("\n" + "=" * 60)
    print("M3 MATCHER TRAINING")
    print("=" * 60)

    print(
        f"Training rows: {len(X):,}"
    )

    print(
        f"Positive pairs: {positive_count:,}"
    )

    print(
        f"Negative pairs: {negative_count:,}"
    )

    if positive_count == 0:
        raise ValueError(
            "No positive training pairs found."
        )

    if negative_count == 0:
        raise ValueError(
            "No negative training pairs found."
        )

    print(
        f"Positive rate: "
        f"{positive_count / len(y):.6%}"
    )

    print(
        f"\nML features: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=2,
        max_features="sqrt",
    )

    print("\nTraining Random Forest...")

    model.fit(
        X,
        y,
    )

    print("Training complete.")

    return model, feature_columns


# ============================================================
# EVALUATE TRAINING MODEL
# ============================================================

def evaluate_matcher(
    model: RandomForestClassifier,
    features: pd.DataFrame,
    training_pairs: pd.DataFrame,
    feature_columns: list[str],
) -> dict:
    """
    Evaluate the matcher on the supplied labeled data.

    This is diagnostic only.
    It is NOT the final M4 threshold selection.
    """

    X, y, _ = prepare_training_data(
        features,
        training_pairs,
    )

    X = X.reindex(
        columns=feature_columns,
        fill_value=0.0,
    )

    probabilities = model.predict_proba(
        X
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {}

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    if len(np.unique(y)) == 2:

        metrics["roc_auc"] = (
            roc_auc_score(
                y,
                probabilities,
            )
        )

        metrics["average_precision"] = (
            average_precision_score(
                y,
                probabilities,
            )
        )

    print("\n" + "=" * 60)
    print("M3 MATCHER DIAGNOSTIC EVALUATION")
    print("=" * 60)

    if "roc_auc" in metrics:

        print(
            f"ROC-AUC: "
            f"{metrics['roc_auc']:.6f}"
        )

        print(
            f"Average Precision: "
            f"{metrics['average_precision']:.6f}"
        )

    print("\nClassification report at threshold 0.50:")

    print(
        classification_report(
            y,
            predictions,
            zero_division=0,
        )
    )

    return metrics


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(
    model: RandomForestClassifier,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Return feature importance ranking.

    This is diagnostic and useful for understanding
    which evidence signals the model relies on.
    """

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    )

    return (
        importance
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================
# PREDICT MATCH PROBABILITIES
# ============================================================

def predict_match_probabilities(
    model: RandomForestClassifier,
    features: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Generate match probabilities for candidate pairs.

    Returns the original candidate identifiers plus:

        match_probability
    """

    X = features[
        feature_columns
    ].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X = X.fillna(0.0)

    probabilities = model.predict_proba(
        X
    )[:, 1]

    result = features[
        [
            "source1_id",
            "candidate_id",
            "candidate_source",
        ]
    ].copy()

    result["match_probability"] = probabilities

    return result


# ============================================================
# SAVE MODEL
# ============================================================

def save_matcher(
    model: RandomForestClassifier,
    feature_columns: list[str],
    path: str | Path = DEFAULT_MODEL_PATH,
) -> Path:
    """
    Save model and feature metadata.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "model": model,
        "feature_columns": feature_columns,
        "model_type": "RandomForestClassifier",
    }

    joblib.dump(
        payload,
        path,
    )

    print(
        f"\nModel saved to:\n{path}"
    )

    return path


# ============================================================
# LOAD MODEL
# ============================================================

def load_matcher(
    path: str | Path = DEFAULT_MODEL_PATH,
) -> tuple[RandomForestClassifier, list[str]]:
    """
    Load a previously saved M3 matcher.
    """

    path = Path(path)

    payload = joblib.load(
        path
    )

    model = payload["model"]

    feature_columns = payload[
        "feature_columns"
    ]

    return model, feature_columns