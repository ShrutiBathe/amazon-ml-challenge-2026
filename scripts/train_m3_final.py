from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

BASE = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(BASE))


from src.matching.model import (
    train_matcher,
    evaluate_matcher,
    get_feature_importance,
    save_matcher,
)


# ============================================================
# PATHS
# ============================================================

FEATURES_PATH = (
    BASE
    / "dataset"
    / "processed"
    / "m3_test"
    / "m3_training_subset_features.tsv"
)

TRAINING_PAIRS_PATH = (
    BASE
    / "dataset"
    / "processed"
    / "m3_test"
    / "m3_training_subset.tsv"
)

MODEL_PATH = (
    BASE
    / "dataset"
    / "processed"
    / "m3_test"
    / "m3_matcher_final.joblib"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("M3 FINAL MATCHER TRAINING")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

if not FEATURES_PATH.exists():
    raise FileNotFoundError(
        f"Feature file not found:\n{FEATURES_PATH}"
    )

if not TRAINING_PAIRS_PATH.exists():
    raise FileNotFoundError(
        f"Training-pair file not found:\n"
        f"{TRAINING_PAIRS_PATH}"
    )


# ============================================================
# LOAD FEATURES
# ============================================================

print("\nLoading feature dataset...")

features = pd.read_csv(
    FEATURES_PATH,
    sep="\t",
)

print(
    f"Features loaded: "
    f"{len(features):,} rows × "
    f"{len(features.columns)} columns"
)


# ============================================================
# IMPORTANT:
# model.py expects the LABEL to come from training_pairs.
#
# build_m3_training_features.py also stores label in the
# feature file, so remove that duplicate column before calling
# train_matcher().
#
# Otherwise pandas creates label_x / label_y during merge.
# ============================================================

if "label" in features.columns:

    features = features.drop(
        columns=["label"]
    )

    print(
        "Removed label column from "
        "feature dataset before training."
    )


print(
    f"Features supplied to model: "
    f"{len(features):,} rows × "
    f"{len(features.columns)} columns"
)


# ============================================================
# LOAD TRAINING PAIRS
# ============================================================

print("\nLoading training pairs...")

training_pairs = pd.read_csv(
    TRAINING_PAIRS_PATH,
    sep="\t",
    dtype=str,
)

print(
    f"Training pairs loaded: "
    f"{len(training_pairs):,}"
)


# ============================================================
# VERIFY LABELS
# ============================================================

if "label" not in training_pairs.columns:
    raise ValueError(
        "Training-pair file does not contain "
        "the required 'label' column."
    )

print("\nLabel distribution:")

print(
    training_pairs["label"]
    .value_counts()
)


# ============================================================
# VERIFY IDENTIFIERS
# ============================================================

required_ids = [
    "source1_id",
    "candidate_id",
    "candidate_source",
]

for column in required_ids:

    if column not in features.columns:
        raise ValueError(
            f"Feature dataset is missing "
            f"required column: {column}"
        )

    if column not in training_pairs.columns:
        raise ValueError(
            f"Training pairs are missing "
            f"required column: {column}"
        )


# ============================================================
# TRAIN MATCHER
# ============================================================

print("\n" + "=" * 70)
print("STARTING RANDOM FOREST TRAINING")
print("=" * 70)

model, feature_columns = train_matcher(
    features=features,
    training_pairs=training_pairs,
    n_estimators=300,
    random_state=42,
)


# ============================================================
# DIAGNOSTIC EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("RUNNING DIAGNOSTIC EVALUATION")
print("=" * 70)

metrics = evaluate_matcher(
    model=model,
    features=features,
    training_pairs=training_pairs,
    feature_columns=feature_columns,
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = get_feature_importance(
    model=model,
    feature_columns=feature_columns,
)

print("\n" + "=" * 70)
print("TOP 15 M3 FEATURE IMPORTANCES")
print("=" * 70)

print(
    importance.head(15).to_string(
        index=False
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\n" + "=" * 70)
print("SAVING M3 MODEL")
print("=" * 70)

saved_path = save_matcher(
    model=model,
    feature_columns=feature_columns,
    path=MODEL_PATH,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("M3 TRAINING COMPLETE")
print("=" * 70)

print(
    f"Training rows: "
    f"{len(features):,}"
)

print(
    f"ML features: "
    f"{len(feature_columns)}"
)

print(
    f"Positive pairs: "
    f"{int((training_pairs['label'].astype(int) == 1).sum()):,}"
)

print(
    f"Negative pairs: "
    f"{int((training_pairs['label'].astype(int) == 0).sum()):,}"
)

if "roc_auc" in metrics:

    print(
        f"ROC-AUC: "
        f"{metrics['roc_auc']:.6f}"
    )

if "average_precision" in metrics:

    print(
        f"Average Precision: "
        f"{metrics['average_precision']:.6f}"
    )

print(
    f"\nModel saved to:\n"
    f"{saved_path}"
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)