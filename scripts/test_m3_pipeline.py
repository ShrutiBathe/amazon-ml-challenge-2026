from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

BASE = Path(__file__).resolve().parents[1]

# Allow Python to import src/
sys.path.insert(0, str(BASE))


# ============================================================
# M3 IMPORTS
# ============================================================

from src.matching.safe_candidates import generate_safe_candidates
from src.matching.features import build_pair_features
from src.matching.train_pairs import (
    build_ground_truth_pairs,
    build_training_pairs,
    training_pair_statistics,
)


# ============================================================
# DATA PATHS
# ============================================================

# Processed M1 files
TRAIN_PROCESSED_DIR = (
    BASE
    / "dataset"
    / "processed"
    / "train"
)

SOURCE1 = (
    TRAIN_PROCESSED_DIR
    / "train_source1.tsv"
)

SOURCE2 = (
    TRAIN_PROCESSED_DIR
    / "train_source2.tsv"
)

SOURCE3 = (
    TRAIN_PROCESSED_DIR
    / "train_source3.tsv"
)


# IMPORTANT:
# Ground truth is stored in dataset/train/,
# not dataset/processed/train/.

GROUND_TRUTH = (
    BASE
    / "dataset"
    / "train"
    / "train_ground_truth.tsv"
)


# ============================================================
# M3 TEST OUTPUT DIRECTORY
# ============================================================

TEST_DIR = (
    BASE
    / "dataset"
    / "processed"
    / "m3_test"
)

TEST_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("M3 PIPELINE SMOKE TEST")
print("=" * 70)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

print("\nChecking required files...")

required_files = {
    "Source1": SOURCE1,
    "Source2": SOURCE2,
    "Source3": SOURCE3,
    "Ground Truth": GROUND_TRUTH,
}

for name, path in required_files.items():

    if not path.exists():

        print(
            f"\nERROR: {name} file not found:"
        )

        print(path)

        raise SystemExit(1)

    print(
        f"  OK - {name}: {path}"
    )


# ============================================================
# 1. LOAD SOURCE1 SAMPLE
# ============================================================

print("\n[1/5] Loading Source1 sample...")

source1_sample = pd.read_csv(
    SOURCE1,
    sep="\t",
    dtype=str,
    nrows=1000,
)

source1_sample_path = (
    TEST_DIR
    / "source1_sample.tsv"
)

source1_sample.to_csv(
    source1_sample_path,
    sep="\t",
    index=False,
)

print(
    f"Source1 sample: "
    f"{len(source1_sample):,} rows"
)


# ============================================================
# 2. LOAD SOURCE2 AND SOURCE3 SAMPLES
# ============================================================

print("\n[2/5] Loading candidate samples...")

source2_sample = pd.read_csv(
    SOURCE2,
    sep="\t",
    dtype=str,
    nrows=200_000,
)

source3_sample = pd.read_csv(
    SOURCE3,
    sep="\t",
    dtype=str,
    nrows=200_000,
)


source2_sample_path = (
    TEST_DIR
    / "source2_sample.tsv"
)

source3_sample_path = (
    TEST_DIR
    / "source3_sample.tsv"
)


source2_sample.to_csv(
    source2_sample_path,
    sep="\t",
    index=False,
)

source3_sample.to_csv(
    source3_sample_path,
    sep="\t",
    index=False,
)


print(
    f"Source2 sample: "
    f"{len(source2_sample):,} rows"
)

print(
    f"Source3 sample: "
    f"{len(source3_sample):,} rows"
)


# ============================================================
# 3. GENERATE SAFE CANDIDATES
# ============================================================

print("\n[3/5] Generating safe candidates...")

candidates = generate_safe_candidates(
    source1_path=source1_sample_path,
    source2_path=source2_sample_path,
    source3_path=source3_sample_path,
)


print(
    f"\nTotal candidate pairs: "
    f"{len(candidates):,}"
)


if candidates.empty:

    print(
        "\nWARNING: No candidates generated."
    )

    raise SystemExit(1)


# ------------------------------------------------------------
# Candidates by source
# ------------------------------------------------------------

print("\nCandidates by source:")

print(
    candidates[
        "candidate_source"
    ]
    .value_counts()
    .to_string()
)


# ------------------------------------------------------------
# Candidates by blocking strategy
# ------------------------------------------------------------

print("\nCandidates by strategy:")

print(
    candidates[
        "blocking_strategy"
    ]
    .value_counts()
    .to_string()
)


# ------------------------------------------------------------
# Candidates per Source1 entity
# ------------------------------------------------------------

print("\nCandidates per Source1 entity:")

candidate_counts = (
    candidates
    .groupby("source1_id")
    .size()
)

print(
    candidate_counts
    .describe()
    .to_string()
)


# ------------------------------------------------------------
# Zero-candidate entities
# ------------------------------------------------------------

zero_candidate_entities = (
    len(source1_sample)
    - candidates["source1_id"].nunique()
)

print(
    "\nSource1 entities with zero candidates: "
    f"{zero_candidate_entities:,}"
)


# ============================================================
# 4. BUILD M3 FEATURES
# ============================================================

print("\n[4/5] Building M3 features...")

candidate_sources = {
    "source2": source2_sample,
    "source3": source3_sample,
}


features = build_pair_features(
    source1=source1_sample,
    candidate_sources=candidate_sources,
    candidates=candidates,
)


print(
    f"Feature rows: "
    f"{len(features):,}"
)

print(
    f"Feature columns: "
    f"{len(features.columns):,}"
)


print("\nFeature columns:")

for column in features.columns:

    print(
        f"  - {column}"
    )


# ============================================================
# 5. BUILD TRAINING PAIRS
# ============================================================

print("\n[5/5] Building training labels...")

ground_truth = pd.read_csv(
    GROUND_TRUTH,
    sep="\t",
    dtype=str,
)

print(
    f"Ground-truth rows: "
    f"{len(ground_truth):,}"
)


# ------------------------------------------------------------
# IMPORTANT:
# build_training_pairs() expects the original ground-truth
# DataFrame. It internally converts it into true pairs.
# ------------------------------------------------------------

training_pairs = build_training_pairs(
    candidates=candidates,
    ground_truth=ground_truth,
)


print(
    f"Training pairs: "
    f"{len(training_pairs):,}"
)


# ------------------------------------------------------------
# Training statistics
# ------------------------------------------------------------

stats = training_pair_statistics(
    training_pairs
)


print("\nTraining pair statistics:")

for key, value in stats.items():

    print(
        f"  {key}: {value}"
    )


# ============================================================
# SAVE TEST OUTPUTS
# ============================================================

print("\nSaving M3 smoke-test outputs...")


features_output = (
    TEST_DIR
    / "m3_test_features.tsv"
)

training_output = (
    TEST_DIR
    / "m3_test_training_pairs.tsv"
)


features.to_csv(
    features_output,
    sep="\t",
    index=False,
)


training_pairs.to_csv(
    training_output,
    sep="\t",
    index=False,
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("M3 PIPELINE SMOKE TEST COMPLETE")
print("=" * 70)


print(
    "\nFeatures saved to:"
)

print(features_output)


print(
    "\nTraining pairs saved to:"
)

print(training_output)


print("\nPipeline verified:")

print("  M1 processed data")
print("       ↓")
print("  Safe M2-style candidate generation")
print("       ↓")
print("  M3 feature engineering")
print("       ↓")
print("  Ground-truth labeling")
print("       ↓")
print("  Training pairs")