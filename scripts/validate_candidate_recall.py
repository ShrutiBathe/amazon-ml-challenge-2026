from pathlib import Path
import sys

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.matching.safe_candidates import generate_safe_candidates


# ============================================================
# PATHS
# ============================================================

PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
PROCESSED_TRAIN_DIR = PROCESSED_DIR / "train"
TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"

SOURCE1_PATH = PROCESSED_TRAIN_DIR / "train_source1.tsv"
SOURCE2_PATH = PROCESSED_TRAIN_DIR / "train_source2.tsv"
SOURCE3_PATH = PROCESSED_TRAIN_DIR / "train_source3.tsv"

GROUND_TRUTH_PATH = TRAIN_DIR / "train_ground_truth.tsv"

# ============================================================
# SETTINGS
# ============================================================

MAX_BLOCK_SIZE = 300
CHUNK_SIZE = 100000


# ============================================================
# FILE CHECK
# ============================================================

def check_files():

    print("\nChecking required files...")

    required_files = [
        SOURCE1_PATH,
        SOURCE2_PATH,
        SOURCE3_PATH,
        GROUND_TRUTH_PATH,
    ]

    for path in required_files:

        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

        print(f"FOUND: {path}")


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_true_pairs():

    print("\nLoading ground truth...")

    ground_truth = pd.read_csv(
        GROUND_TRUTH_PATH,
        sep="\t",
        usecols=[
            "source1_entity_id",
            "matched_entity_ids",
        ],
    )

    print(
        f"Ground-truth rows: "
        f"{len(ground_truth):,}"
    )

    true_pairs = set()

    for _, row in ground_truth.iterrows():

        source1_id = str(
            row["source1_entity_id"]
        ).strip()

        matched_ids = row["matched_entity_ids"]

        if pd.isna(matched_ids):
            continue

        matched_ids = str(
            matched_ids
        ).strip()

        if not matched_ids:
            continue

        for candidate_id in matched_ids.split(","):

            candidate_id = candidate_id.strip()

            if not candidate_id:
                continue

            if candidate_id.startswith("S2-"):
                candidate_source = "source2"

            elif candidate_id.startswith("S3-"):
                candidate_source = "source3"

            else:
                continue

            true_pairs.add(
                (
                    source1_id,
                    candidate_id,
                    candidate_source,
                )
            )

    print(
        f"Total ground-truth true pairs: "
        f"{len(true_pairs):,}"
    )

    return true_pairs


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("M3 CANDIDATE RECALL VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Check files
    # --------------------------------------------------------

    check_files()

    # --------------------------------------------------------
    # 2. Load ground truth
    # --------------------------------------------------------

    true_pairs = load_true_pairs()

    if not true_pairs:

        print(
            "\nERROR: No true pairs were extracted "
            "from ground truth."
        )

        return

    # --------------------------------------------------------
    # 3. Generate candidates
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GENERATING CANDIDATES")
    print("=" * 70)

    print(
        f"\nMaximum block size : "
        f"{MAX_BLOCK_SIZE:,}"
    )

    print(
        f"Chunk size         : "
        f"{CHUNK_SIZE:,}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This function loads the complete Source1 "
        "training dataset."
    )

    print(
        "The validation therefore evaluates candidate "
        "recall over the complete training Source1."
    )

    candidates = generate_safe_candidates(
        source1_path=SOURCE1_PATH,
        source2_path=SOURCE2_PATH,
        source3_path=SOURCE3_PATH,
        max_block_size=MAX_BLOCK_SIZE,
        chunk_size=CHUNK_SIZE,
    )

    print(
        f"\nGenerated candidate pairs: "
        f"{len(candidates):,}"
    )

    # --------------------------------------------------------
    # 4. Normalize columns
    # --------------------------------------------------------

    required_columns = {
        "source1_id",
        "candidate_id",
        "candidate_source",
    }

    missing = required_columns - set(
        candidates.columns
    )

    if missing:

        raise ValueError(
            f"Candidate output is missing columns: "
            f"{missing}"
        )

    candidates["source1_id"] = (
        candidates["source1_id"]
        .astype(str)
        .str.strip()
    )

    candidates["candidate_id"] = (
        candidates["candidate_id"]
        .astype(str)
        .str.strip()
    )

    candidates["candidate_source"] = (
        candidates["candidate_source"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # 5. Save candidate pairs
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates.to_csv(
        CANDIDATES_OUTPUT,
        sep="\t",
        index=False,
    )

    print(
        f"\nCandidate pairs saved to:"
    )

    print(CANDIDATES_OUTPUT)

    # --------------------------------------------------------
    # 6. Convert candidates to set
    # --------------------------------------------------------

    print(
        "\nBuilding candidate pair lookup..."
    )

    candidate_pairs = set(
        zip(
            candidates["source1_id"],
            candidates["candidate_id"],
            candidates["candidate_source"],
        )
    )

    # --------------------------------------------------------
    # 7. Calculate recall
    # --------------------------------------------------------

    recovered = true_pairs.intersection(
        candidate_pairs
    )

    missed = true_pairs - candidate_pairs

    recall = (
        len(recovered) / len(true_pairs)
        if true_pairs
        else 0.0
    )

    # --------------------------------------------------------
    # 8. Results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CANDIDATE RECALL RESULTS")
    print("=" * 70)

    print(
        f"\nGround-truth true pairs : "
        f"{len(true_pairs):,}"
    )

    print(
        f"Recovered true pairs    : "
        f"{len(recovered):,}"
    )

    print(
        f"Missed true pairs       : "
        f"{len(missed):,}"
    )

    print(
        f"\nCandidate Recall        : "
        f"{recall * 100:.4f}%"
    )

    # --------------------------------------------------------
    # 9. Source-wise recall
    # --------------------------------------------------------

    s2_true = {
        pair for pair in true_pairs
        if pair[2] == "source2"
    }

    s3_true = {
        pair for pair in true_pairs
        if pair[2] == "source3"
    }

    s2_recovered = s2_true.intersection(
        candidate_pairs
    )

    s3_recovered = s3_true.intersection(
        candidate_pairs
    )

    print("\nSource-wise recall:")

    if s2_true:

        print(
            f"Source2: "
            f"{len(s2_recovered):,} / "
            f"{len(s2_true):,} "
            f"= "
            f"{len(s2_recovered) / len(s2_true) * 100:.4f}%"
        )

    if s3_true:

        print(
            f"Source3: "
            f"{len(s3_recovered):,} / "
            f"{len(s3_true):,} "
            f"= "
            f"{len(s3_recovered) / len(s3_true) * 100:.4f}%"
        )

    # --------------------------------------------------------
    # 10. Missed examples
    # --------------------------------------------------------

    if missed:

        print(
            "\nFirst 20 missed true pairs:"
        )

        for source1_id, candidate_id, source in list(
            missed
        )[:20]:

            print(
                f"  {source1_id} -> "
                f"{candidate_id} ({source})"
            )

    else:

        print(
            "\nNo ground-truth pairs were missed."
        )

    # --------------------------------------------------------
    # 11. Candidate statistics
    # --------------------------------------------------------

    candidates_per_source1 = (
        candidates
        .groupby("source1_id")
        .size()
    )

    print("\nCandidate statistics:")

    print(
        f"Source1 entities with candidates : "
        f"{len(candidates_per_source1):,}"
    )

    print(
        f"Mean candidates per Source1      : "
        f"{candidates_per_source1.mean():.2f}"
    )

    print(
        f"Median candidates per Source1    : "
        f"{candidates_per_source1.median():.2f}"
    )

    print(
        f"Maximum candidates per Source1   : "
        f"{candidates_per_source1.max():,}"
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CANDIDATE RECALL VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()