from pathlib import Path
import sys

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Make src/ importable when running:
# python scripts\test_m3_model.py
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import pandas as pd

from src.matching.model import (
    train_matcher,
    predict_match_probabilities,
    get_feature_importance,
)


# ============================================================
# FILE PATHS
# ============================================================

FEATURES_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "m3_test"
    / "m3_test_features.tsv"
)

TRAINING_PAIRS_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "m3_test"
    / "m3_test_training_pairs.tsv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("M3 ML MATCHER SMOKE TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    print("\nChecking required files...")

    if not FEATURES_PATH.exists():
        print("\nERROR: Features file not found:")
        print(FEATURES_PATH)
        return

    if not TRAINING_PAIRS_PATH.exists():
        print("\nERROR: Training pairs file not found:")
        print(TRAINING_PAIRS_PATH)
        return

    print("Features file: FOUND")
    print("Training pairs file: FOUND")

    # --------------------------------------------------------
    # 1. Load features
    # --------------------------------------------------------

    print("\n[1/5] Loading M3 features...")

    features = pd.read_csv(
        FEATURES_PATH,
        sep="\t"
    )

    print(f"Features shape: {features.shape}")

    print("\nFeature columns:")
    print(list(features.columns))

    # --------------------------------------------------------
    # 2. Load training pairs
    # --------------------------------------------------------

    print("\n[2/5] Loading training pairs...")

    training_pairs = pd.read_csv(
        TRAINING_PAIRS_PATH,
        sep="\t"
    )

    print(f"Training pairs shape: {training_pairs.shape}")

    if "label" not in training_pairs.columns:
        print("\nERROR: 'label' column is missing.")
        return

    print("\nLabel distribution:")
    print(training_pairs["label"].value_counts())

    # --------------------------------------------------------
    # 3. Train Random Forest matcher
    # --------------------------------------------------------

    print("\n[3/5] Training Random Forest matcher...")

    model, feature_columns = train_matcher(
        features,
        training_pairs,
        n_estimators=100,
        random_state=42,
    )

    print("\nModel trained successfully.")

    print(f"Number of ML features: {len(feature_columns)}")

    print("\nML feature columns:")
    for i, feature in enumerate(feature_columns, start=1):
        print(f"{i:02d}. {feature}")

    # --------------------------------------------------------
    # 4. Generate match probabilities
    # --------------------------------------------------------

    print("\n[4/5] Generating match probabilities...")

    predictions = predict_match_probabilities(
        model,
        features,
        feature_columns,
    )

    print("\nPrediction columns:")
    print(list(predictions.columns))

    print("\nFirst 10 predictions:")
    print(
        predictions.head(10).to_string(index=False)
    )

    print("\nProbability statistics:")

    print(
        predictions["match_probability"].describe()
    )

    # --------------------------------------------------------
    # 5. Feature importance
    # --------------------------------------------------------

    print("\n[5/5] Checking feature importance...")

    importance = get_feature_importance(
        model,
        feature_columns,
    )

    print("\nTop 15 important features:")

    print(
        importance.head(15).to_string(index=False)
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("M3 MODEL SMOKE TEST COMPLETE")
    print("=" * 70)

    print("\nThe following pipeline is working:")
    print("M3 Features")
    print("     ↓")
    print("Training pairs + labels")
    print("     ↓")
    print("Random Forest matcher")
    print("     ↓")
    print("Match probabilities")
    print("     ↓")
    print("Feature importance")

    print("\n")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()