import pandas as pd

TRAIN_S1 = "dataset/processed/train/train_source1.tsv"
TRAIN_S2 = "dataset/processed/train/train_source2.tsv"
TRAIN_S3 = "dataset/processed/train/train_source3.tsv"
GROUND_TRUTH = "dataset/train/train_ground_truth.tsv"


def main():
    print("Loading Source 1 sample...")
    s1 = pd.read_csv(
        TRAIN_S1,
        sep="\t",
        nrows=10000,
    )

    print("Loading Source 2...")
    s2 = pd.read_csv(
        TRAIN_S2,
        sep="\t",
    )

    print("Loading Source 3...")
    s3 = pd.read_csv(
        TRAIN_S3,
        sep="\t",
    )

    print("Loading ground truth...")
    gt = pd.read_csv(
        GROUND_TRUTH,
        sep="\t",
    )

    # Only evaluate S1 records in our 10k sample
    gt = gt[
        gt["source1_entity_id"].isin(
            s1["entity_id"]
        )
    ]

    # Fast lookup by entity ID
    s1i = s1.set_index("entity_id")
    s2i = s2.set_index("entity_id")
    s3i = s3.set_index("entity_id")

    stats = {
        "total": 0,
        "name_prefix": 0,
        "first_token": 0,
        "name_no_suffix": 0,
        "address_first_token": 0,
        "country": 0,
    }

    print(
        f"Ground-truth S1 records: "
        f"{len(gt):,}"
    )

    print("Analyzing true matches...")

    for _, row in gt.iterrows():

        s1_id = row["source1_entity_id"]

        if s1_id not in s1i.index:
            continue

        source1 = s1i.loc[s1_id]

        matched_ids = str(
            row["matched_entity_ids"]
        ).split(",")

        for candidate_id in matched_ids:

            candidate_id = candidate_id.strip()

            if candidate_id.startswith("S2-"):
                if candidate_id not in s2i.index:
                    continue

                candidate = s2i.loc[candidate_id]

            elif candidate_id.startswith("S3-"):
                if candidate_id not in s3i.index:
                    continue

                candidate = s3i.loc[candidate_id]

            else:
                continue

            stats["total"] += 1

            if (
                source1["name_prefix_6"]
                == candidate["name_prefix_6"]
            ):
                stats["name_prefix"] += 1

            if (
                source1["name_first_token"]
                == candidate["name_first_token"]
            ):
                stats["first_token"] += 1

            if (
                source1["name_no_suffix_compact"]
                == candidate["name_no_suffix_compact"]
            ):
                stats["name_no_suffix"] += 1

            if (
                source1["address_first_token"]
                == candidate["address_first_token"]
            ):
                stats["address_first_token"] += 1

            if (
                source1["country_normalized"]
                == candidate["country_normalized"]
            ):
                stats["country"] += 1

    print()
    print("=" * 70)
    print("GROUND-TRUTH BLOCKING KEY ANALYSIS")
    print("=" * 70)

    print(
        f"Total true matches analyzed: "
        f"{stats['total']:,}"
    )

    print()

    for key, value in stats.items():

        if key == "total":
            continue

        percentage = (
            value / stats["total"] * 100
            if stats["total"] > 0
            else 0
        )

        print(
            f"{key:<25} "
            f"{value:>10,} "
            f"({percentage:>6.2f}%)"
        )

    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    print(
        """
These percentages show how often a blocking key
is shared by records that are known to match.

A high percentage means the key has potential
for high-recall blocking.

A low percentage means the key alone would miss
many true matches.

IMPORTANT:
This measures RECALL POTENTIAL only.
It does not measure candidate-set size or precision.
"""
    )


if __name__ == "__main__":
    main()