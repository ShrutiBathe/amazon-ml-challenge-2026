from pathlib import Path
from collections import Counter

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

GROUND_TRUTH = (
    BASE_DIR
    / "dataset"
    / "train"
    / "train_ground_truth.tsv"
)


def main():

    print("=" * 70)
    print("GROUND TRUTH ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Read ground truth
    # --------------------------------------------------------

    gt = pd.read_csv(
        GROUND_TRUTH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(f"Rows: {len(gt):,}")
    print(f"Columns: {list(gt.columns)}")

    required = [
        "source1_entity_id",
        "matched_entity_ids",
    ]

    for col in required:
        if col not in gt.columns:
            raise ValueError(
                f"Missing required column: {col}"
            )

    # --------------------------------------------------------
    # Match cardinality
    # --------------------------------------------------------

    def count_matches(value):

        if not value or not value.strip():
            return 0

        return len([
            x for x in value.split(",")
            if x.strip()
        ])

    match_counts = (
    gt["matched_entity_ids"]
    .map(count_matches)
    )

    gt = gt.assign(match_count=match_counts)

    counts = gt["match_count"].value_counts().sort_index()

    print()
    print("=" * 70)
    print("MATCH CARDINALITY")
    print("=" * 70)

    for count, rows in counts.items():

        percentage = (
            rows / len(gt) * 100
        )

        print(
            f"{count:>3} matches : "
            f"{rows:>10,} entities "
            f"({percentage:6.2f}%)"
        )

    # --------------------------------------------------------
    # Singleton / multi-match statistics
    # --------------------------------------------------------

    zero = (gt["match_count"] == 0).sum()
    one = (gt["match_count"] == 1).sum()
    multiple = (gt["match_count"] > 1).sum()

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"No match       : {zero:,}")
    print(f"Single match   : {one:,}")
    print(f"Multiple match : {multiple:,}")

    # --------------------------------------------------------
    # Matched entity source distribution
    # --------------------------------------------------------

    source_counter = Counter()

    total_matched_ids = 0

    for value in gt["matched_entity_ids"]:

        if not value:
            continue

        ids = [
            x.strip()
            for x in value.split(",")
            if x.strip()
        ]

        for entity_id in ids:

            total_matched_ids += 1

            if entity_id.startswith("S2-"):
                source_counter["S2"] += 1

            elif entity_id.startswith("S3-"):
                source_counter["S3"] += 1

            else:
                source_counter["OTHER"] += 1

    print()
    print("=" * 70)
    print("MATCHED ENTITY SOURCE DISTRIBUTION")
    print("=" * 70)

    print(
        f"Total matched IDs: "
        f"{total_matched_ids:,}"
    )

    for source, count in source_counter.items():

        percentage = (
            count / total_matched_ids * 100
            if total_matched_ids
            else 0
        )

        print(
            f"{source}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # Examples
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EXAMPLES")
    print("=" * 70)

    print("\nNo-match examples:")

    print(
        gt.loc[
            gt["match_count"] == 0,
            [
                "source1_entity_id",
                "matched_entity_ids",
            ],
        ]
        .head(5)
        .to_string(index=False)
    )

    print("\nSingle-match examples:")

    print(
        gt.loc[
            gt["match_count"] == 1,
            [
                "source1_entity_id",
                "matched_entity_ids",
            ],
        ]
        .head(5)
        .to_string(index=False)
    )

    print("\nMulti-match examples:")

    print(
        gt.loc[
            gt["match_count"] > 1,
            [
                "source1_entity_id",
                "matched_entity_ids",
            ],
        ]
        .head(5)
        .to_string(index=False)
    )

    print()
    print("=" * 70)
    print("GROUND TRUTH ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()