from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]

GT_PATH = (
    BASE
    / "dataset"
    / "train"
    / "train_ground_truth.tsv"
)

CANDIDATE_PATH = (
    BASE
    / "dataset"
    / "processed"
    / "m3_candidates"
    / "train_candidate_pairs.tsv"
)

print("=" * 70)
print("COUNTING TRUE MATCHES INSIDE CANDIDATE SET")
print("=" * 70)

# ------------------------------------------------------------
# Load ground truth
# ------------------------------------------------------------

print("\nLoading ground truth...")

gt = pd.read_csv(
    GT_PATH,
    sep="\t",
    usecols=[
        "source1_entity_id",
        "matched_entity_ids",
    ],
)

print(
    f"Ground-truth rows: {len(gt):,}"
)

# ------------------------------------------------------------
# Build lookup
# ------------------------------------------------------------

print("\nBuilding ground-truth lookup...")

gt_map = {}

for _, row in gt.iterrows():

    source1_id = str(
        row["source1_entity_id"]
    ).strip()

    matched = row["matched_entity_ids"]

    if pd.isna(matched):
        continue

    matched_ids = str(
        matched
    ).split(",")

    gt_map[source1_id] = {
        x.strip()
        for x in matched_ids
        if x.strip()
    }

print(
    f"Source1 entities in GT: "
    f"{len(gt_map):,}"
)

# ------------------------------------------------------------
# Scan candidate file
# ------------------------------------------------------------

print("\nScanning candidate pairs...")

total_candidates = 0
true_candidates = 0

for chunk_number, chunk in enumerate(
    pd.read_csv(
        CANDIDATE_PATH,
        sep="\t",
        dtype=str,
        chunksize=200_000,
    ),
    start=1,
):

    total_candidates += len(chunk)

    for source1_id, candidate_id in zip(
        chunk["source1_id"],
        chunk["candidate_id"],
    ):

        source1_id = str(
            source1_id
        ).strip()

        candidate_id = str(
            candidate_id
        ).strip()

        if candidate_id in gt_map.get(
            source1_id,
            set(),
        ):
            true_candidates += 1

    print(
        f"Chunk {chunk_number}: "
        f"{total_candidates:,} candidates | "
        f"{true_candidates:,} true"
    )

# ------------------------------------------------------------
# Final result
# ------------------------------------------------------------

positive_rate = (
    true_candidates
    / total_candidates
    * 100
    if total_candidates
    else 0
)

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(
    f"\nTotal candidates       : "
    f"{total_candidates:,}"
)

print(
    f"True candidate pairs   : "
    f"{true_candidates:,}"
)

print(
    f"Candidate positive rate: "
    f"{positive_rate:.4f}%"
)

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)