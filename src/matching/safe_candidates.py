from collections import Counter
from pathlib import Path

import pandas as pd

CANDIDATE_COLUMNS = [
    "source1_id",
    "candidate_id",
    "candidate_source",
    "blocking_strategy",
]

MAX_BLOCK_SIZE = 300
CHUNK_SIZE = 100_000


def _valid_keys(
    candidate_path,
    column,
    chunk_size=CHUNK_SIZE,
):
    """
    Count candidate-side blocking keys.
    Only keys occurring <= MAX_BLOCK_SIZE times are retained.
    """

    counts = Counter()

    for chunk in pd.read_csv(
        candidate_path,
        sep="\t",
        dtype=str,
        usecols=["entity_id", column],
        chunksize=chunk_size,
    ):
        values = (
            chunk[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        values = values[values != ""]

        counts.update(values.tolist())

    return {
        key
        for key, count in counts.items()
        if count <= MAX_BLOCK_SIZE
    }


def _generate_one_strategy(
    source1,
    candidate_path,
    candidate_source,
    column,
    strategy,
    output_path,
    chunk_size=CHUNK_SIZE,
):
    """
    Vectorized candidate generation for one blocking strategy.
    """

    print(
        f"\n  {candidate_source} | "
        f"{strategy} | {column}"
    )

    if column not in source1.columns:
        print("    Column missing.")
        return 0

    print("    Finding valid blocking keys...")

    valid_keys = _valid_keys(
        candidate_path,
        column,
        chunk_size,
    )

    print(
        f"    Valid keys: {len(valid_keys):,}"
    )

    if not valid_keys:
        return 0

    s1 = source1[
        ["entity_id", column]
    ].copy()

    s1[column] = (
        s1[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    s1 = s1[
        s1[column].isin(valid_keys)
    ]

    if s1.empty:
        return 0

    s1 = s1.rename(
        columns={
            "entity_id": "source1_id"
        }
    )

    total = 0
    first_write = True

    for chunk in pd.read_csv(
        candidate_path,
        sep="\t",
        dtype=str,
        usecols=["entity_id", column],
        chunksize=chunk_size,
    ):

        chunk[column] = (
            chunk[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        chunk = chunk[
            chunk[column].isin(valid_keys)
        ]

        if chunk.empty:
            continue

        chunk = chunk.rename(
            columns={
                "entity_id": "candidate_id"
            }
        )

        merged = s1.merge(
            chunk,
            on=column,
            how="inner",
        )

        if merged.empty:
            continue

        result = merged[
            [
                "source1_id",
                "candidate_id",
            ]
        ].copy()

        result["candidate_source"] = (
            candidate_source
        )

        result["blocking_strategy"] = (
            strategy
        )

        result = result[
            CANDIDATE_COLUMNS
        ]

        result.to_csv(
            output_path,
            sep="\t",
            index=False,
            mode="w" if first_write else "a",
            header=first_write,
        )

        first_write = False

        total += len(result)

    print(
        f"    Generated: {total:,}"
    )

    return total


def generate_emergency_candidates(
    source1_path,
    source2_path,
    source3_path,
    output_path,
):
    source1_path = Path(source1_path)
    source2_path = Path(source2_path)
    source3_path = Path(source3_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists():
        output_path.unlink()

    print("=" * 70)
    print("EMERGENCY VECTORISED CANDIDATE GENERATION")
    print("=" * 70)

    print("\nLoading Source1...")

    source1 = pd.read_csv(
        source1_path,
        sep="\t",
        dtype=str,
        usecols=[
            "entity_id",
            "name_compact",
            "name_prefix_6",
            "address_first_token",
            "name_first_token",
        ],
    )

    print(
        f"Source1 rows: {len(source1):,}"
    )

    # --------------------------------------------------------
    # ONLY THE THREE MOST IMPORTANT ROUTES
    # --------------------------------------------------------

    strategies = [
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
    ]

    total = 0

    for candidate_source, candidate_path in [
        ("source2", source2_path),
        ("source3", source3_path),
    ]:

        print("\n" + "=" * 70)
        print(
            f"PROCESSING {candidate_source.upper()}"
        )
        print("=" * 70)

        for strategy, column in strategies:

            total += _generate_one_strategy(
                source1=source1,
                candidate_path=candidate_path,
                candidate_source=candidate_source,
                column=column,
                strategy=strategy,
                output_path=output_path,
            )

    print("\n" + "=" * 70)
    print("REMOVING DUPLICATES")
    print("=" * 70)

    # Use pandas chunked deduplication.
    temp_path = output_path.with_suffix(
        ".tmp.tsv"
    )

    if temp_path.exists():
        temp_path.unlink()

    seen = set()
    first_write = True

    for chunk in pd.read_csv(
        output_path,
        sep="\t",
        dtype=str,
        chunksize=200_000,
    ):

        chunk = chunk.drop_duplicates(
            subset=[
                "source1_id",
                "candidate_source",
                "candidate_id",
            ]
        )

        keys = list(
            zip(
                chunk["source1_id"],
                chunk["candidate_source"],
                chunk["candidate_id"],
            )
        )

        keep = []

        for key in keys:

            if key in seen:
                keep.append(False)
            else:
                seen.add(key)
                keep.append(True)

        chunk = chunk.loc[keep]

        if chunk.empty:
            continue

        chunk.to_csv(
            temp_path,
            sep="\t",
            index=False,
            mode="w" if first_write else "a",
            header=first_write,
        )

        first_write = False

    output_path.unlink()
    temp_path.rename(output_path)

    final_count = 0

    for chunk in pd.read_csv(
        output_path,
        sep="\t",
        dtype=str,
        chunksize=200_000,
    ):
        final_count += len(chunk)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)

    print(
        f"Final unique candidates: "
        f"{final_count:,}"
    )

    print(
        f"Output: {output_path}"
    )


if __name__ == "__main__":

    BASE = Path(__file__).resolve().parents[2]

    TRAIN_DIR = (
        BASE
        / "dataset"
        / "processed"
        / "train"
    )

    OUTPUT_DIR = (
        BASE
        / "dataset"
        / "processed"
        / "m3_candidates"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    generate_emergency_candidates(
        source1_path=(
            TRAIN_DIR
            / "train_source1.tsv"
        ),
        source2_path=(
            TRAIN_DIR
            / "train_source2.tsv"
        ),
        source3_path=(
            TRAIN_DIR
            / "train_source3.tsv"
        ),
        output_path=(
            OUTPUT_DIR
            / "train_candidate_pairs.tsv"
        ),
    )