from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

BASE = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(BASE))

from src.matching.features import build_pair_features


# ============================================================
# PATHS
# ============================================================

TRAIN_DIR = (
    BASE
    / "dataset"
    / "processed"
    / "train"
)

INPUT_PATH = (
    BASE
    / "dataset"
    / "processed"
    / "m3_test"
    / "m3_training_subset.tsv"
)

OUTPUT_PATH = (
    BASE
    / "dataset"
    / "processed"
    / "m3_test"
    / "m3_training_subset_features.tsv"
)

SOURCE1_PATH = (
    TRAIN_DIR
    / "train_source1.tsv"
)

SOURCE2_PATH = (
    TRAIN_DIR
    / "train_source2.tsv"
)

SOURCE3_PATH = (
    TRAIN_DIR
    / "train_source3.tsv"
)


# ============================================================
# SETTINGS
# ============================================================

PAIR_CHUNK_SIZE = 25_000
SOURCE_CHUNK_SIZE = 100_000


# ============================================================
# REQUIRED COLUMNS
# ============================================================

USECOLS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
    "name_normalized",
    "name_compact",
    "name_no_suffix",
    "name_no_suffix_compact",
    "name_dedup",
    "address_normalized",
    "address_compact",
    "country_normalized",
    "name_prefix",
    "name_prefix_6",
    "name_first_token",
    "address_first_token",
]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("M3 TRAINING FEATURE GENERATION")
print("=" * 70)


# ============================================================
# CHECK INPUT
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Training subset not found:\n{INPUT_PATH}"
    )


# ============================================================
# FIND REQUIRED IDs
# ============================================================

print("\nReading training-pair IDs...")

required_s1 = set()
required_s2 = set()
required_s3 = set()

for chunk_number, chunk in enumerate(
    pd.read_csv(
        INPUT_PATH,
        sep="\t",
        dtype=str,
        usecols=[
            "source1_id",
            "candidate_id",
            "candidate_source",
            "label",
        ],
        chunksize=PAIR_CHUNK_SIZE,
    ),
    start=1,
):

    required_s1.update(
        chunk["source1_id"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    source2_mask = (
        chunk["candidate_source"]
        .astype(str)
        .str.strip()
        == "source2"
    )

    source3_mask = (
        chunk["candidate_source"]
        .astype(str)
        .str.strip()
        == "source3"
    )

    required_s2.update(
        chunk.loc[
            source2_mask,
            "candidate_id",
        ]
        .dropna()
        .astype(str)
        .str.strip()
    )

    required_s3.update(
        chunk.loc[
            source3_mask,
            "candidate_id",
        ]
        .dropna()
        .astype(str)
        .str.strip()
    )


print(
    f"Required Source1 IDs: "
    f"{len(required_s1):,}"
)

print(
    f"Required Source2 IDs: "
    f"{len(required_s2):,}"
)

print(
    f"Required Source3 IDs: "
    f"{len(required_s3):,}"
)


# ============================================================
# LOAD REQUIRED RECORDS
# ============================================================

def load_required(path, required_ids, label):

    print(
        f"\nLoading required {label} records..."
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # If there are no IDs for this source, don't call
    # pd.concat([]), because that causes:
    #
    # ValueError: No objects to concatenate
    # --------------------------------------------------------

    if not required_ids:

        print(
            f"No {label} IDs required; "
            f"skipping load."
        )

        return pd.DataFrame(
            columns=USECOLS
        )

    parts = []

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            usecols=USECOLS,
            chunksize=SOURCE_CHUNK_SIZE,
        ),
        start=1,
    ):

        chunk["entity_id"] = (
            chunk["entity_id"]
            .astype(str)
            .str.strip()
        )

        mask = (
            chunk["entity_id"]
            .isin(required_ids)
        )

        if mask.any():

            parts.append(
                chunk.loc[
                    mask
                ].copy()
            )

        if chunk_number % 10 == 0:

            print(
                f"  Scanned "
                f"{chunk_number} chunks..."
            )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if not parts:

        print(
            f"WARNING: No matching {label} "
            f"records were found."
        )

        return pd.DataFrame(
            columns=USECOLS
        )

    result = pd.concat(
        parts,
        ignore_index=True,
    )

    result = result.drop_duplicates(
        subset=["entity_id"]
    )

    print(
        f"{label} loaded: "
        f"{len(result):,}"
    )

    return result


# ============================================================
# LOAD SOURCE1
# ============================================================

source1 = load_required(
    SOURCE1_PATH,
    required_s1,
    "Source1",
)


# ============================================================
# LOAD SOURCE2
# ============================================================

source2 = load_required(
    SOURCE2_PATH,
    required_s2,
    "Source2",
)


# ============================================================
# LOAD SOURCE3
# ============================================================

source3 = load_required(
    SOURCE3_PATH,
    required_s3,
    "Source3",
)


# ============================================================
# CANDIDATE SOURCE DICTIONARY
# ============================================================

candidate_sources = {
    "source2": source2,
    "source3": source3,
}


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("LOADED DATA SUMMARY")
print("=" * 70)

print(
    f"Source1 records: "
    f"{len(source1):,}"
)

print(
    f"Source2 records: "
    f"{len(source2):,}"
)

print(
    f"Source3 records: "
    f"{len(source3):,}"
)

if source1.empty:
    raise RuntimeError(
        "Source1 is empty. "
        "Cannot generate pair features."
    )

if source2.empty and source3.empty:
    raise RuntimeError(
        "Both Source2 and Source3 are empty. "
        "Cannot generate pair features."
    )


# ============================================================
# PREPARE OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("GENERATING M3 FEATURES")
print("=" * 70)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

if OUTPUT_PATH.exists():
    OUTPUT_PATH.unlink()

first_write = True
total = 0


# ============================================================
# GENERATE FEATURES CHUNK BY CHUNK
# ============================================================

for chunk_number, candidates in enumerate(
    pd.read_csv(
        INPUT_PATH,
        sep="\t",
        dtype=str,
        chunksize=PAIR_CHUNK_SIZE,
    ),
    start=1,
):

    print(
        f"\nChunk {chunk_number}: "
        f"{len(candidates):,} pairs"
    )

    # --------------------------------------------------------
    # Preserve labels
    # --------------------------------------------------------

    labels = (
        candidates["label"]
        .astype(str)
        .copy()
    )

    # --------------------------------------------------------
    # Only columns required by build_pair_features()
    # --------------------------------------------------------

    candidates_for_features = candidates[
        [
            "source1_id",
            "candidate_id",
            "candidate_source",
        ]
    ].copy()

    # --------------------------------------------------------
    # Generate M3 features
    # --------------------------------------------------------

    features = build_pair_features(
        source1=source1,
        candidate_sources=candidate_sources,
        candidates=candidates_for_features,
    )

    # --------------------------------------------------------
    # Restore label
    # --------------------------------------------------------

    features["label"] = labels.values

    # --------------------------------------------------------
    # Write chunk
    # --------------------------------------------------------

    features.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False,
        mode=(
            "w"
            if first_write
            else "a"
        ),
        header=first_write,
    )

    first_write = False

    total += len(features)

    print(
        f"  Generated features: "
        f"{len(features):,}"
    )

    print(
        f"  Total processed: "
        f"{total:,}"
    )


# ============================================================
# FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)

if not OUTPUT_PATH.exists():
    raise RuntimeError(
        "Feature output file was not created."
    )


# Read only metadata columns first
verification = pd.read_csv(
    OUTPUT_PATH,
    sep="\t",
    dtype=str,
    usecols=["label"],
)

print(
    f"Feature rows: "
    f"{len(verification):,}"
)

print(
    f"Feature columns: "
    f"{len(pd.read_csv(OUTPUT_PATH, sep='\t', nrows=1).columns):,}"
)

print("\nLabel distribution:")

print(
    verification["label"]
    .value_counts()
)


print("\nOutput:")

print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("M3 FEATURE GENERATION COMPLETE")
print("=" * 70)