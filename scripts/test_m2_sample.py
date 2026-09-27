from pathlib import Path
import sys

import pandas as pd


# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

BASE = Path(__file__).resolve().parents[1]

# Allow Python to import src/
sys.path.insert(0, str(BASE))


# ------------------------------------------------------------
# M2 IMPORT
# ------------------------------------------------------------

from src.blocking.blocking import create_v4_engine


# ------------------------------------------------------------
# DATA PATHS
# ------------------------------------------------------------

TRAIN_DIR = BASE / "dataset" / "processed" / "train"

SOURCE1 = TRAIN_DIR / "train_source1.tsv"
SOURCE2 = TRAIN_DIR / "train_source2.tsv"
SOURCE3 = TRAIN_DIR / "train_source3.tsv"


# ------------------------------------------------------------
# LOAD SOURCE1 SAMPLE
# ------------------------------------------------------------

print("=" * 60)
print("M2 V4.1 SAMPLE TEST")
print("=" * 60)

print("\nLoading Source1 sample...")

source1 = pd.read_csv(
    SOURCE1,
    sep="\t",
    nrows=5000,
    dtype=str,
)

print(f"Source1 sample rows: {len(source1):,}")


# ------------------------------------------------------------
# LOAD SOURCE2
# ------------------------------------------------------------

print("\nLoading Source2...")

source2 = pd.read_csv(
    SOURCE2,
    sep="\t",
    dtype=str,
)

print(f"Source2 rows: {len(source2):,}")


# ------------------------------------------------------------
# LOAD SOURCE3
# ------------------------------------------------------------

print("\nLoading Source3...")

source3 = pd.read_csv(
    SOURCE3,
    sep="\t",
    dtype=str,
)

print(f"Source3 rows: {len(source3):,}")


# ------------------------------------------------------------
# CANDIDATE SOURCES
# ------------------------------------------------------------

candidate_sources = {
    "source2": source2,
    "source3": source3,
}


# ------------------------------------------------------------
# CREATE M2 ENGINE
# ------------------------------------------------------------

print("\nCreating M2 V4.1 engine...")

engine = create_v4_engine()

print("M2 V4.1 engine created.")

print("\nBlocking strategies:")

for name, _ in engine.strategies:
    print(f"  - {name}")


# ------------------------------------------------------------
# GENERATE CANDIDATES
# ------------------------------------------------------------

print("\nGenerating candidates...")
print("Please wait...")

candidates = engine.generate_candidates(
    source1,
    candidate_sources,
)


# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("M2 SAMPLE RESULT")
print("=" * 60)

print(f"Source1 rows:    {len(source1):,}")
print(f"Candidate pairs: {len(candidates):,}")


if candidates.empty:

    print("\nNo candidates generated.")

else:

    # --------------------------------------------------------
    # CANDIDATES BY SOURCE
    # --------------------------------------------------------

    print("\nCandidates by source:")

    print(
        candidates["candidate_source"]
        .value_counts()
        .to_string()
    )


    # --------------------------------------------------------
    # CANDIDATES BY BLOCKING STRATEGY
    # --------------------------------------------------------

    print("\nCandidates by blocking strategy:")

    print(
        candidates["blocking_strategy"]
        .value_counts()
        .to_string()
    )


    # --------------------------------------------------------
    # CANDIDATES PER SOURCE1 ENTITY
    # --------------------------------------------------------

    print("\nCandidates per Source1 entity:")

    candidate_counts = (
        candidates
        .groupby("source1_id")
        .size()
    )

    print(
        candidate_counts.describe()
        .to_string()
    )


    # --------------------------------------------------------
    # ZERO-CANDIDATE SOURCE1 ENTITIES
    # --------------------------------------------------------

    zero_candidates = (
        len(source1)
        - candidates["source1_id"].nunique()
    )

    print(
        f"\nSource1 entities with zero candidates: "
        f"{zero_candidates:,}"
    )


    # --------------------------------------------------------
    # SAVE SAMPLE CANDIDATES
    # --------------------------------------------------------

    output = (
        TRAIN_DIR
        / "m2_sample_candidates.tsv"
    )

    candidates.to_csv(
        output,
        sep="\t",
        index=False,
    )

    print(f"\nSample candidates saved to:")
    print(output)


# ------------------------------------------------------------
# FINISHED
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("M2 SAMPLE TEST COMPLETE")
print("=" * 60)