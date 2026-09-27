"""
M3 - Training Pair Construction

Builds supervised training pairs from:
    1. M2 candidate pairs
    2. Training ground truth

Positive:
    M2 candidate exists in ground truth.

Hard negative:
    M2 candidate does not exist in ground truth.

The important design choice is that negatives come from M2 candidates.
Therefore they are realistic near-matches that already survived blocking.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


GROUND_TRUTH_COLUMNS = {
    "source1_entity_id",
    "matched_entity_ids",
}

CANDIDATE_COLUMNS = {
    "source1_id",
    "candidate_id",
    "candidate_source",
}


def build_ground_truth_pairs(
    ground_truth: pd.DataFrame,
) -> set[tuple[str, str]]:
    """
    Convert ground truth into unique (source1_id, candidate_id) pairs.

    Example:
        S1-123 -> S2-456,S2-789,S3-111

    becomes:
        (S1-123, S2-456)
        (S1-123, S2-789)
        (S1-123, S3-111)
    """

    missing = GROUND_TRUTH_COLUMNS - set(ground_truth.columns)

    if missing:
        raise ValueError(
            f"Missing ground-truth columns: {sorted(missing)}"
        )

    true_pairs: set[tuple[str, str]] = set()

    for _, row in ground_truth.iterrows():

        source1_id = str(
            row["source1_entity_id"]
        ).strip()

        if not source1_id:
            continue

        matched_value = row["matched_entity_ids"]

        if pd.isna(matched_value):
            continue

        matched_ids = str(
            matched_value
        ).split(",")

        for matched_id in matched_ids:

            matched_id = matched_id.strip()

            if matched_id:
                true_pairs.add(
                    (source1_id, matched_id)
                )

    return true_pairs


def build_training_pairs(
    candidates: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    """
    Label M2 candidates using training ground truth.

    Returns one row per unique candidate pair.

    Columns:
        source1_id
        candidate_id
        candidate_source
        label

    label:
        1 = true match
        0 = hard negative
    """

    missing = CANDIDATE_COLUMNS - set(candidates.columns)

    if missing:
        raise ValueError(
            f"Missing candidate columns: {sorted(missing)}"
        )

    true_pairs = build_ground_truth_pairs(
        ground_truth
    )

    candidate_df = candidates.copy()

    candidate_df["source1_id"] = (
        candidate_df["source1_id"]
        .astype(str)
        .str.strip()
    )

    candidate_df["candidate_id"] = (
        candidate_df["candidate_id"]
        .astype(str)
        .str.strip()
    )

    candidate_df["candidate_source"] = (
        candidate_df["candidate_source"]
        .astype(str)
        .str.strip()
    )

    # Remove exact duplicate candidate rows.
    candidate_df = candidate_df.drop_duplicates(
        subset=[
            "source1_id",
            "candidate_id",
            "candidate_source",
        ]
    ).reset_index(drop=True)

    candidate_df["pair_key"] = list(
        zip(
            candidate_df["source1_id"],
            candidate_df["candidate_id"],
        )
    )

    candidate_df["label"] = candidate_df[
        "pair_key"
    ].isin(true_pairs).astype(int)

    candidate_df = candidate_df.drop(
        columns=["pair_key"]
    )

    return candidate_df


def training_pair_statistics(
    training_pairs: pd.DataFrame,
) -> dict[str, Any]:
    """
    Calculate useful statistics for the generated training pairs.
    """

    required = {
        "source1_id",
        "candidate_id",
        "candidate_source",
        "label",
    }

    missing = required - set(training_pairs.columns)

    if missing:
        raise ValueError(
            f"Missing training-pair columns: {sorted(missing)}"
        )

    total = len(training_pairs)

    positives = int(
        (training_pairs["label"] == 1).sum()
    )

    negatives = int(
        (training_pairs["label"] == 0).sum()
    )

    positive_rate = (
        positives / total
        if total > 0
        else 0.0
    )

    source_statistics = (
        training_pairs
        .groupby(
            ["candidate_source", "label"]
        )
        .size()
        .to_dict()
    )

    return {
        "total_pairs": total,
        "positive_pairs": positives,
        "negative_pairs": negatives,
        "positive_rate": positive_rate,
        "source_statistics": source_statistics,
    }


def print_training_pair_statistics(
    training_pairs: pd.DataFrame,
) -> None:
    """Print human-readable training-pair statistics."""

    stats = training_pair_statistics(
        training_pairs
    )

    print("\n" + "=" * 60)
    print("M3 TRAINING PAIR STATISTICS")
    print("=" * 60)

    print(
        f"Total candidate pairs : "
        f"{stats['total_pairs']}"
    )

    print(
        f"Positive pairs        : "
        f"{stats['positive_pairs']}"
    )

    print(
        f"Hard negative pairs   : "
        f"{stats['negative_pairs']}"
    )

    print(
        f"Positive rate         : "
        f"{stats['positive_rate']:.4f}"
    )

    print("\nSource / label distribution:")

    for key, value in sorted(
        stats["source_statistics"].items()
    ):
        source, label = key

        label_name = (
            "positive"
            if label == 1
            else "negative"
        )

        print(
            f"  {source:10s} "
            f"{label_name:10s}: {value}"
        )


__all__ = [
    "build_ground_truth_pairs",
    "build_training_pairs",
    "training_pair_statistics",
    "print_training_pair_statistics",
]