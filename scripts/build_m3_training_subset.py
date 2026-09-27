from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]

CANDIDATE_PATH = (
    BASE
    / "dataset"
    / "processed"
    / "m3_candidates"
    / "train_candidate_pairs.tsv"
)

GT_PATH = (
    BASE
    / "dataset"
    / "train"
    / "train_ground_truth.tsv"
)

OUTPUT_PATH = (
    BASE
    / "dataset"
    / "processed"
    / "m3_test"
    / "m3_training_subset.tsv"
)

CHUNK_SIZE = 200_000
RANDOM_STATE = 42

# Balanced across candidate source
POSITIVE_PER_SOURCE = 50_000
NEGATIVE_PER_SOURCE = 150_000


print("=" * 70)
print("BUILDING BALANCED M3 TRAINING SUBSET")
print("=" * 70)


# ============================================================
# Load ground truth
# ============================================================

print("\nLoading ground truth...")

gt = pd.read_csv(
    GT_PATH,
    sep="\t",
    usecols=[
        "source1_entity_id",
        "matched_entity_ids",
    ],
)

true_pairs = set()

for _, row in gt.iterrows():

    source1_id = str(
        row["source1_entity_id"]
    ).strip()

    matched = row["matched_entity_ids"]

    if pd.isna(matched):
        continue

    for candidate_id in str(matched).split(","):

        candidate_id = candidate_id.strip()

        if candidate_id:
            true_pairs.add(
                (source1_id, candidate_id)
            )

print(
    f"Ground-truth pairs: "
    f"{len(true_pairs):,}"
)


# ============================================================
# Collect Source2 / Source3 separately
# ============================================================

positive_parts = {
    "source2": [],
    "source3": [],
}

negative_parts = {
    "source2": [],
    "source3": [],
}

positive_counts = {
    "source2": 0,
    "source3": 0,
}

negative_counts = {
    "source2": 0,
    "source3": 0,
}


print("\nScanning candidate file...")

for chunk_number, chunk in enumerate(
    pd.read_csv(
        CANDIDATE_PATH,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
    ),
    start=1,
):

    chunk["pair_key"] = list(
        zip(
            chunk["source1_id"],
            chunk["candidate_id"],
        )
    )

    positive_mask = chunk[
        "pair_key"
    ].isin(true_pairs)

    for source in ["source2", "source3"]:

        source_mask = (
            chunk["candidate_source"]
            == source
        )

        source_chunk = chunk.loc[
            source_mask
        ].copy()

        if source_chunk.empty:
            continue

        source_positive = source_chunk.loc[
            source_chunk["pair_key"].isin(
                true_pairs
            )
        ].drop(
            columns=["pair_key"]
        )

        source_negative = source_chunk.loc[
            ~source_chunk["pair_key"].isin(
                true_pairs
            )
        ].drop(
            columns=["pair_key"]
        )

        if not source_positive.empty:

            positive_parts[source].append(
                source_positive
            )

            positive_counts[source] += (
                len(source_positive)
            )

        if not source_negative.empty:

            negative_parts[source].append(
                source_negative
            )

            negative_counts[source] += (
                len(source_negative)
            )

    print(
        f"Chunk {chunk_number}: "
        f"S2 +{positive_counts['source2']:,}/"
        f"-{negative_counts['source2']:,} | "
        f"S3 +{positive_counts['source3']:,}/"
        f"-{negative_counts['source3']:,}"
    )


# ============================================================
# Combine + sample
# ============================================================

selected_parts = []

for source in ["source2"]:

    print("\n" + "-" * 60)
    print(source.upper())

    positives = pd.concat(
        positive_parts[source],
        ignore_index=True,
    )

    negatives = pd.concat(
        negative_parts[source],
        ignore_index=True,
    )

    print(
        f"Available positives: "
        f"{len(positives):,}"
    )

    print(
        f"Available negatives: "
        f"{len(negatives):,}"
    )

    positive_n = min(
        POSITIVE_PER_SOURCE,
        len(positives),
    )

    negative_n = min(
        NEGATIVE_PER_SOURCE,
        len(negatives),
    )

    positives = positives.sample(
        n=positive_n,
        random_state=RANDOM_STATE,
    )

    negatives = negatives.sample(
        n=negative_n,
        random_state=RANDOM_STATE,
    )

    positives["label"] = 1
    negatives["label"] = 0

    selected_parts.extend(
        [
            positives,
            negatives,
        ]
    )

    print(
        f"Selected positives: "
        f"{positive_n:,}"
    )

    print(
        f"Selected negatives: "
        f"{negative_n:,}"
    )


# ============================================================
# Final dataset
# ============================================================

training = pd.concat(
    selected_parts,
    ignore_index=True,
)

training = training.sample(
    frac=1,
    random_state=RANDOM_STATE,
).reset_index(drop=True)


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

training.to_csv(
    OUTPUT_PATH,
    sep="\t",
    index=False,
)


# ============================================================
# Final report
# ============================================================

print("\n" + "=" * 70)
print("BALANCED TRAINING SUBSET CREATED")
print("=" * 70)

print(
    f"\nTotal pairs: "
    f"{len(training):,}"
)

print("\nSource distribution:")
print(
    training[
        "candidate_source"
    ].value_counts()
)

print("\nSource + label distribution:")
print(
    training
    .groupby(
        [
            "candidate_source",
            "label",
        ]
    )
    .size()
)

print(
    f"\nOutput:\n{OUTPUT_PATH}"
)

print("\nDONE")