from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# DATASET PATHS
# ============================================================

TRAIN_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "train"
)

SOURCE1_PATH = TRAIN_DIR / "train_source1.tsv"
SOURCE2_PATH = TRAIN_DIR / "train_source2.tsv"
SOURCE3_PATH = TRAIN_DIR / "train_source3.tsv"

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "train"
    / "train_ground_truth.tsv"
)


# ============================================================
# SETTINGS
# ============================================================

# Number of true pairs used for this validation.
# We are NOT generating all candidate pairs.
SAMPLE_TRUE_PAIRS = 50_000

RANDOM_STATE = 42

# Chunk size for reading large TSV files.
CHUNK_SIZE = 100_000

# Same blocking limit currently used by M2/safe candidate logic.
MAX_BLOCK_SIZE = 300


# ============================================================
# BLOCKING STRATEGIES
# ============================================================

STRATEGIES = [
    ("exact_name", "name_compact"),
    ("name_prefix", "name_prefix_6"),
    ("address_token", "address_first_token"),
    ("first_token", "name_first_token"),
    ("address_first_token", "address_first_token"),
]


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

def check_required_files():

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
                f"\nRequired file not found:\n{path}"
            )

        print(f"FOUND: {path}")


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth_sample():

    print("\n" + "=" * 70)
    print("LOADING GROUND TRUTH")
    print("=" * 70)

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

    records = []

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

                records.append(
                    (
                        source1_id,
                        candidate_id,
                        "source2",
                    )
                )

            elif candidate_id.startswith("S3-"):

                records.append(
                    (
                        source1_id,
                        candidate_id,
                        "source3",
                    )
                )

    true_pairs = pd.DataFrame(
        records,
        columns=[
            "source1_id",
            "candidate_id",
            "candidate_source",
        ],
    )

    # Remove accidental duplicate ground-truth pairs.
    true_pairs = (
        true_pairs
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(
        f"Total true pairs: "
        f"{len(true_pairs):,}"
    )

    if len(true_pairs) > SAMPLE_TRUE_PAIRS:

        true_pairs = (
            true_pairs
            .sample(
                n=SAMPLE_TRUE_PAIRS,
                random_state=RANDOM_STATE,
            )
            .reset_index(drop=True)
        )

    print(
        f"Validation sample: "
        f"{len(true_pairs):,}"
    )

    print("\nSource distribution:")

    print(
        true_pairs[
            "candidate_source"
        ].value_counts()
    )

    return true_pairs


# ============================================================
# LOAD REQUIRED SOURCE1 RECORDS
# ============================================================

def load_required_source1(source1_ids):

    print("\n" + "=" * 70)
    print("LOADING REQUIRED SOURCE1 RECORDS")
    print("=" * 70)

    source1_ids = set(
        str(x).strip()
        for x in source1_ids
    )

    required_columns = [
        "entity_id",
        "name_compact",
        "name_prefix_6",
        "address_first_token",
        "name_first_token",
    ]

    parts = []

    chunk_number = 0

    for chunk in pd.read_csv(
        SOURCE1_PATH,
        sep="\t",
        usecols=required_columns,
        chunksize=CHUNK_SIZE,
        dtype=str,
    ):

        chunk_number += 1

        ids = (
            chunk["entity_id"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        mask = ids.isin(source1_ids)

        if mask.any():

            parts.append(
                chunk.loc[mask].copy()
            )

        if chunk_number % 10 == 0:

            print(
                f"  Source1 chunks scanned: "
                f"{chunk_number}"
            )

    if not parts:

        raise RuntimeError(
            "No required Source1 records were found."
        )

    result = pd.concat(
        parts,
        ignore_index=True,
    )

    result["entity_id"] = (
        result["entity_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Ensure one row per entity.
    result = (
        result
        .drop_duplicates(
            subset=["entity_id"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    print(
        f"\nSource1 records loaded: "
        f"{len(result):,}"
    )

    return result


# ============================================================
# LOAD REQUIRED CANDIDATE RECORDS
# ============================================================

def load_required_candidates(
    path,
    candidate_ids,
    source_name,
):

    print("\n" + "=" * 70)

    print(
        f"LOADING REQUIRED {source_name.upper()} RECORDS"
    )

    print("=" * 70)

    candidate_ids = set(
        str(x).strip()
        for x in candidate_ids
    )

    required_columns = [
        "entity_id",
        "name_compact",
        "name_prefix_6",
        "address_first_token",
        "name_first_token",
    ]

    parts = []

    chunk_number = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=required_columns,
        chunksize=CHUNK_SIZE,
        dtype=str,
    ):

        chunk_number += 1

        ids = (
            chunk["entity_id"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        mask = ids.isin(candidate_ids)

        if mask.any():

            parts.append(
                chunk.loc[mask].copy()
            )

        if chunk_number % 10 == 0:

            print(
                f"  {source_name} chunks scanned: "
                f"{chunk_number}"
            )

    if not parts:

        raise RuntimeError(
            f"No required {source_name} records found."
        )

    result = pd.concat(
        parts,
        ignore_index=True,
    )

    result["entity_id"] = (
        result["entity_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    result = (
        result
        .drop_duplicates(
            subset=["entity_id"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    print(
        f"\n{source_name} records loaded: "
        f"{len(result):,}"
    )

    return result


# ============================================================
# CHECK RECALL FOR ONE SOURCE
# ============================================================

def check_source_recall(
    true_pairs,
    source1_df,
    candidate_df,
    candidate_source,
):

    print("\n" + "=" * 70)

    print(
        f"CHECKING {candidate_source.upper()} RECALL"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Select true pairs for this source.
    # --------------------------------------------------------

    pairs = (
        true_pairs[
            true_pairs["candidate_source"]
            == candidate_source
        ]
        .copy()
        .reset_index(drop=True)
    )

    if pairs.empty:

        print(
            f"No {candidate_source} true pairs."
        )

        return {
            "total": 0,
            "recovered": 0,
            "missed": 0,
            "recall": 0.0,
            "strategy_hits": {},
        }

    print(
        f"True pairs: "
        f"{len(pairs):,}"
    )

    # --------------------------------------------------------
    # Prepare Source1.
    # --------------------------------------------------------

    source1 = source1_df[
        [
            "entity_id",
            "name_compact",
            "name_prefix_6",
            "address_first_token",
            "name_first_token",
        ]
    ].copy()

    source1 = (
        source1
        .drop_duplicates(
            subset=["entity_id"],
            keep="first",
        )
        .rename(
            columns={
                "entity_id": "source1_id",
                "name_compact": "s1_name_compact",
                "name_prefix_6": "s1_name_prefix_6",
                "address_first_token": "s1_address_first_token",
                "name_first_token": "s1_name_first_token",
            }
        )
    )

    # --------------------------------------------------------
    # Prepare candidate.
    # --------------------------------------------------------

    candidate = candidate_df[
        [
            "entity_id",
            "name_compact",
            "name_prefix_6",
            "address_first_token",
            "name_first_token",
        ]
    ].copy()

    candidate = (
        candidate
        .drop_duplicates(
            subset=["entity_id"],
            keep="first",
        )
        .rename(
            columns={
                "entity_id": "candidate_id",
                "name_compact": "candidate_name_compact",
                "name_prefix_6": "candidate_name_prefix_6",
                "address_first_token": "candidate_address_first_token",
                "name_first_token": "candidate_name_first_token",
            }
        )
    )

    # --------------------------------------------------------
    # Merge true pairs with their Source1 data.
    # --------------------------------------------------------

    pairs = pairs.merge(
        source1,
        on="source1_id",
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # Merge true pairs with candidate data.
    # --------------------------------------------------------

    pairs = pairs.merge(
        candidate,
        on="candidate_id",
        how="left",
        validate="many_to_one",
    )

    # IMPORTANT:
    # Always reset the index after merge.
    pairs = pairs.reset_index(drop=True)

    print(
        f"Merged true pairs: "
        f"{len(pairs):,}"
    )

    # --------------------------------------------------------
    # Calculate candidate-side block sizes.
    # --------------------------------------------------------

    block_sizes = {}

    for strategy_name, column in STRATEGIES:

        counts = (
            candidate_df[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .value_counts()
        )

        block_sizes[column] = counts

    # --------------------------------------------------------
    # Boolean recovery array.
    #
    # This is a NumPy-backed boolean array through pandas,
    # but we convert it to a plain list only at the end.
    # --------------------------------------------------------

    recovered = pd.Series(
        False,
        index=range(len(pairs)),
        dtype=bool,
    )

    strategy_hits = {}

    # --------------------------------------------------------
    # Strategy 1: exact_name
    # --------------------------------------------------------

    strategy_definitions = [
        (
            "exact_name",
            "s1_name_compact",
            "candidate_name_compact",
            "name_compact",
        ),
        (
            "name_prefix",
            "s1_name_prefix_6",
            "candidate_name_prefix_6",
            "name_prefix_6",
        ),
        (
            "address_token",
            "s1_address_first_token",
            "candidate_address_first_token",
            "address_first_token",
        ),
        (
            "first_token",
            "s1_name_first_token",
            "candidate_name_first_token",
            "name_first_token",
        ),
        (
            "address_first_token",
            "s1_address_first_token",
            "candidate_address_first_token",
            "address_first_token",
        ),
    ]

    for (
        strategy_name,
        left_column,
        right_column,
        block_column,
    ) in strategy_definitions:

        print(
            f"\n  Checking strategy: "
            f"{strategy_name}"
        )

        left = (
            pairs[left_column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        right = (
            pairs[right_column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        same_key = (
            left.ne("")
            & right.ne("")
            & left.eq(right)
        )

        # ----------------------------------------------------
        # Block size for the Source2/Source3 side.
        # ----------------------------------------------------

        counts = block_sizes[
            block_column
        ]

        block_size = (
            right.map(counts)
            .fillna(0)
        )

        safe_block = (
            block_size
            <= MAX_BLOCK_SIZE
        )

        hit = (
            same_key
            & safe_block
        )

        new_hit = (
            hit
            & ~recovered
        )

        hit_count = int(
            new_hit.sum()
        )

        strategy_hits[
            strategy_name
        ] = hit_count

        recovered = (
            recovered
            | hit
        )

        print(
            f"    New recoveries: "
            f"{hit_count:,}"
        )

    # --------------------------------------------------------
    # Final result.
    # --------------------------------------------------------

    recovered_count = int(
        recovered.sum()
    )

    total = len(pairs)

    missed_count = (
        total
        - recovered_count
    )

    recall = (
        recovered_count / total
        if total
        else 0.0
    )

    print("\nSource result:")

    print(
        f"  Total true pairs : "
        f"{total:,}"
    )

    print(
        f"  Recovered        : "
        f"{recovered_count:,}"
    )

    print(
        f"  Missed           : "
        f"{missed_count:,}"
    )

    print(
        f"  Recall           : "
        f"{recall * 100:.4f}%"
    )

    print("\nStrategy contribution:")

    for strategy_name in (
        strategy_hits
    ):

        print(
            f"  {strategy_name:25s}: "
            f"{strategy_hits[strategy_name]:,}"
        )

    # --------------------------------------------------------
    # Missed examples.
    # --------------------------------------------------------

    if missed_count > 0:

        missed_pairs = pairs.loc[
            ~recovered,
            [
                "source1_id",
                "candidate_id",
                "candidate_source",
            ],
        ]

        print(
            "\nFirst 20 missed pairs:"
        )

        for _, row in missed_pairs.head(20).iterrows():

            print(
                f"  {row['source1_id']} -> "
                f"{row['candidate_id']} "
                f"({row['candidate_source']})"
            )

    else:

        print(
            "\nNo missed true pairs for "
            f"{candidate_source}."
        )

    return {
        "total": total,
        "recovered": recovered_count,
        "missed": missed_count,
        "recall": recall,
        "strategy_hits": strategy_hits,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FAST CANDIDATE RECALL VALIDATION")
    print("=" * 70)

    print(
        f"\nTrue-pair sample size: "
        f"{SAMPLE_TRUE_PAIRS:,}"
    )

    print(
        f"Maximum block size: "
        f"{MAX_BLOCK_SIZE}"
    )

    # --------------------------------------------------------
    # Check files.
    # --------------------------------------------------------

    check_required_files()

    # --------------------------------------------------------
    # Load ground truth.
    # --------------------------------------------------------

    true_pairs = (
        load_ground_truth_sample()
    )

    if true_pairs.empty:

        raise RuntimeError(
            "No ground-truth pairs were loaded."
        )

    # --------------------------------------------------------
    # Source1 IDs required.
    # --------------------------------------------------------

    source1_ids = set(
        true_pairs["source1_id"]
    )

    # --------------------------------------------------------
    # Load required Source1 records.
    # --------------------------------------------------------

    source1_df = load_required_source1(
        source1_ids
    )

    # --------------------------------------------------------
    # Source2 IDs.
    # --------------------------------------------------------

    source2_pairs = true_pairs[
        true_pairs["candidate_source"]
        == "source2"
    ]

    source2_ids = set(
        source2_pairs["candidate_id"]
    )

    # --------------------------------------------------------
    # Load Source2 records.
    # --------------------------------------------------------

    source2_df = load_required_candidates(
        SOURCE2_PATH,
        source2_ids,
        "source2",
    )

    # --------------------------------------------------------
    # Validate Source2.
    # --------------------------------------------------------

    source2_result = (
        check_source_recall(
            true_pairs,
            source1_df,
            source2_df,
            "source2",
        )
    )

    # --------------------------------------------------------
    # Source3 IDs.
    # --------------------------------------------------------

    source3_pairs = true_pairs[
        true_pairs["candidate_source"]
        == "source3"
    ]

    source3_ids = set(
        source3_pairs["candidate_id"]
    )

    # --------------------------------------------------------
    # Load Source3.
    # --------------------------------------------------------

    source3_df = load_required_candidates(
        SOURCE3_PATH,
        source3_ids,
        "source3",
    )

    # --------------------------------------------------------
    # Validate Source3.
    # --------------------------------------------------------

    source3_result = (
        check_source_recall(
            true_pairs,
            source1_df,
            source3_df,
            "source3",
        )
    )

    # --------------------------------------------------------
    # Overall result.
    # --------------------------------------------------------

    total_pairs = (
        source2_result["total"]
        + source3_result["total"]
    )

    total_recovered = (
        source2_result["recovered"]
        + source3_result["recovered"]
    )

    total_missed = (
        source2_result["missed"]
        + source3_result["missed"]
    )

    overall_recall = (
        total_recovered / total_pairs
        if total_pairs
        else 0.0
    )

    # --------------------------------------------------------
    # Final report.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL RECALL RESULT")
    print("=" * 70)

    print(
        f"\nTotal sampled true pairs : "
        f"{total_pairs:,}"
    )

    print(
        f"Recovered                : "
        f"{total_recovered:,}"
    )

    print(
        f"Missed                   : "
        f"{total_missed:,}"
    )

    print(
        f"\nCANDIDATE RECALL         : "
        f"{overall_recall * 100:.4f}%"
    )

    # --------------------------------------------------------
    # Source-specific results.
    # --------------------------------------------------------

    print("\nSource2:")

    if source2_result["total"]:

        print(
            f"  Total    : "
            f"{source2_result['total']:,}"
        )

        print(
            f"  Recovered: "
            f"{source2_result['recovered']:,}"
        )

        print(
            f"  Missed   : "
            f"{source2_result['missed']:,}"
        )

        print(
            f"  Recall   : "
            f"{source2_result['recall'] * 100:.4f}%"
        )

    print("\nSource3:")

    if source3_result["total"]:

        print(
            f"  Total    : "
            f"{source3_result['total']:,}"
        )

        print(
            f"  Recovered: "
            f"{source3_result['recovered']:,}"
        )

        print(
            f"  Missed   : "
            f"{source3_result['missed']:,}"
        )

        print(
            f"  Recall   : "
            f"{source3_result['recall'] * 100:.4f}%"
        )

    # --------------------------------------------------------
    # Final status.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FAST RECALL VALIDATION COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()