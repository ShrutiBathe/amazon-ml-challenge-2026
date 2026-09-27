from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PATHS
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

SAMPLE_TRUE_PAIRS = 50_000

RANDOM_STATE = 42

CHUNK_SIZE = 100_000

MAX_BLOCK_SIZE = 300


# ============================================================
# CURRENT + EXTENDED BLOCKING STRATEGIES
# ============================================================

STRATEGIES = [

    # --------------------------------------------------------
    # Existing M2 strategies
    # --------------------------------------------------------

    (
        "exact_name",
        "name_compact",
    ),

    (
        "name_prefix",
        "name_prefix_6",
    ),

    (
        "address_token",
        "address_first_token",
    ),

    (
        "first_token",
        "name_first_token",
    ),

    # --------------------------------------------------------
    # Additional M1 fields
    # --------------------------------------------------------

    (
        "name_no_suffix_compact",
        "name_no_suffix_compact",
    ),

    (
        "name_dedup",
        "name_dedup",
    ),

    (
        "address_compact",
        "address_compact",
    ),

    (
        "name_no_suffix",
        "name_no_suffix",
    ),
]


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth():

    print("\nLoading ground truth...")

    gt = pd.read_csv(
        GROUND_TRUTH_PATH,
        sep="\t",
        usecols=[
            "source1_entity_id",
            "matched_entity_ids",
        ],
    )

    print(
        f"Ground-truth rows: "
        f"{len(gt):,}"
    )

    records = []

    for _, row in gt.iterrows():

        source1_id = str(
            row["source1_entity_id"]
        ).strip()

        matched = row["matched_entity_ids"]

        if pd.isna(matched):
            continue

        matched = str(matched).strip()

        if not matched:
            continue

        for candidate_id in matched.split(","):

            candidate_id = candidate_id.strip()

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

    pairs = pd.DataFrame(
        records,
        columns=[
            "source1_id",
            "candidate_id",
            "candidate_source",
        ],
    )

    pairs = (
        pairs
        .drop_duplicates()
        .sample(
            n=min(
                SAMPLE_TRUE_PAIRS,
                len(pairs),
            ),
            random_state=RANDOM_STATE,
        )
        .reset_index(drop=True)
    )

    print(
        f"Validation true pairs: "
        f"{len(pairs):,}"
    )

    print("\nSource distribution:")

    print(
        pairs["candidate_source"].value_counts()
    )

    return pairs


# ============================================================
# LOAD RECORDS
# ============================================================

def load_required_records(
    path,
    entity_ids,
    source_name,
):

    print(
        f"\nLoading required {source_name} records..."
    )

    required_columns = [
        "entity_id",
        "name_compact",
        "name_prefix_6",
        "address_first_token",
        "name_first_token",
        "name_no_suffix",
        "name_no_suffix_compact",
        "name_dedup",
        "address_compact",
    ]

    entity_ids = set(
        str(x).strip()
        for x in entity_ids
    )

    parts = []

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            path,
            sep="\t",
            usecols=required_columns,
            chunksize=CHUNK_SIZE,
            dtype=str,
        ),
        start=1,
    ):

        ids = (
            chunk["entity_id"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        mask = ids.isin(entity_ids)

        if mask.any():

            parts.append(
                chunk.loc[mask].copy()
            )

        if chunk_number % 10 == 0:

            print(
                f"  Chunks scanned: "
                f"{chunk_number}"
            )

    if not parts:

        raise RuntimeError(
            f"No records found for {source_name}"
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
        f"{source_name} records loaded: "
        f"{len(result):,}"
    )

    return result


# ============================================================
# EVALUATE STRATEGIES
# ============================================================

def evaluate_source(
    pairs,
    source1,
    candidate,
    source_name,
):

    pairs = (
        pairs[
            pairs["candidate_source"]
            == source_name
        ]
        .copy()
        .reset_index(drop=True)
    )

    if pairs.empty:

        return None

    print("\n" + "=" * 70)

    print(
        f"ANALYZING {source_name.upper()}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Prepare Source1
    # --------------------------------------------------------

    s1_columns = [
        "entity_id"
    ] + [
        column
        for _, column in STRATEGIES
    ]

    s1 = (
        source1[s1_columns]
        .drop_duplicates(
            subset=["entity_id"]
        )
        .rename(
            columns={
                "entity_id": "source1_id"
            }
        )
    )

    # --------------------------------------------------------
    # Prepare candidate source
    # --------------------------------------------------------

    candidate_columns = [
        "entity_id"
    ] + [
        column
        for _, column in STRATEGIES
    ]

    cand = (
        candidate[candidate_columns]
        .drop_duplicates(
            subset=["entity_id"]
        )
        .rename(
            columns={
                "entity_id": "candidate_id"
            }
        )
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    pairs = pairs.merge(
        s1,
        on="source1_id",
        how="left",
        validate="many_to_one",
    )

    pairs = pairs.merge(
        cand,
        on="candidate_id",
        how="left",
        suffixes=(
            "_s1",
            "_cand",
        ),
        validate="many_to_one",
    )

    pairs = pairs.reset_index(
        drop=True
    )

    recovered = pd.Series(
        False,
        index=pairs.index,
    )

    results = []

    # --------------------------------------------------------
    # Test every strategy
    # --------------------------------------------------------

    for strategy_name, column in STRATEGIES:

        left_column = (
            f"{column}_s1"
        )

        right_column = (
            f"{column}_cand"
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
        # Candidate-side block sizes
        # ----------------------------------------------------

        counts = (
            candidate[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .value_counts()
        )

        block_size = (
            right.map(counts)
            .fillna(0)
        )

        safe_block = (
            block_size
            <= MAX_BLOCK_SIZE
        )

        valid_hit = (
            same_key
            & safe_block
        )

        new_hit = (
            valid_hit
            & ~recovered
        )

        total_hit = int(
            valid_hit.sum()
        )

        new_recovery = int(
            new_hit.sum()
        )

        results.append(
            {
                "strategy": strategy_name,
                "same_key": total_hit,
                "new_recovery": new_recovery,
            }
        )

        recovered |= valid_hit

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    result_df = pd.DataFrame(
        results
    )

    print("\nStrategy results:")

    print(
        result_df.to_string(
            index=False
        )
    )

    recovered_count = int(
        recovered.sum()
    )

    missed_count = (
        len(pairs)
        - recovered_count
    )

    recall = (
        recovered_count
        / len(pairs)
    )

    print("\nCombined result:")

    print(
        f"Total pairs : "
        f"{len(pairs):,}"
    )

    print(
        f"Recovered   : "
        f"{recovered_count:,}"
    )

    print(
        f"Missed      : "
        f"{missed_count:,}"
    )

    print(
        f"Recall      : "
        f"{recall * 100:.4f}%"
    )

    return {
        "total": len(pairs),
        "recovered": recovered_count,
        "missed": missed_count,
        "recall": recall,
        "results": result_df,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("EXTENDED BLOCKING RECALL ANALYSIS")
    print("=" * 70)

    print(
        "\nCurrent block size: "
        f"{MAX_BLOCK_SIZE}"
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    for path in [
        SOURCE1_PATH,
        SOURCE2_PATH,
        SOURCE3_PATH,
        GROUND_TRUTH_PATH,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing file:\n{path}"
            )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    pairs = load_ground_truth()

    # --------------------------------------------------------
    # Source1
    # --------------------------------------------------------

    source1_ids = set(
        pairs["source1_id"]
    )

    source1 = load_required_records(
        SOURCE1_PATH,
        source1_ids,
        "Source1",
    )

    # --------------------------------------------------------
    # Source2
    # --------------------------------------------------------

    source2_ids = set(
        pairs.loc[
            pairs["candidate_source"]
            == "source2",
            "candidate_id",
        ]
    )

    source2 = load_required_records(
        SOURCE2_PATH,
        source2_ids,
        "Source2",
    )

    result2 = evaluate_source(
        pairs,
        source1,
        source2,
        "source2",
    )

    # --------------------------------------------------------
    # Source3
    # --------------------------------------------------------

    source3_ids = set(
        pairs.loc[
            pairs["candidate_source"]
            == "source3",
            "candidate_id",
        ]
    )

    source3 = load_required_records(
        SOURCE3_PATH,
        source3_ids,
        "Source3",
    )

    result3 = evaluate_source(
        pairs,
        source1,
        source3,
        "source3",
    )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    total = (
        result2["total"]
        + result3["total"]
    )

    recovered = (
        result2["recovered"]
        + result3["recovered"]
    )

    missed = (
        result2["missed"]
        + result3["missed"]
    )

    recall = (
        recovered / total
    )

    print("\n" + "=" * 70)
    print("EXTENDED BLOCKING RECALL RESULT")
    print("=" * 70)

    print(
        f"\nTotal true pairs : "
        f"{total:,}"
    )

    print(
        f"Recovered        : "
        f"{recovered:,}"
    )

    print(
        f"Missed           : "
        f"{missed:,}"
    )

    print(
        f"\nEXTENDED RECALL  : "
        f"{recall * 100:.4f}%"
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()