
"""
Validation utilities for M2 blocking.

M2 evaluates:
1. Candidate-pair schema
2. Candidate recall against ground truth
3. Candidate-set size
4. Zero-candidate Source-1 records
"""

from __future__ import annotations

from typing import Any

import pandas as pd


# ============================================================
# BASIC CANDIDATE VALIDATION
# ============================================================

def validate_candidates(candidates: pd.DataFrame) -> None:
    """Validate the basic candidate-pair schema."""

    required_columns = {
        "source1_id",
        "candidate_id",
        "candidate_source",
        "blocking_strategy",
    }

    missing = required_columns - set(candidates.columns)

    if missing:
        raise ValueError(
            f"Missing required candidate columns: {sorted(missing)}"
        )

    if candidates.empty:
        print("WARNING: Candidate dataframe is empty.")
        return

    duplicate_count = candidates.duplicated(
        subset=[
            "source1_id",
            "candidate_source",
            "candidate_id",
        ]
    ).sum()

    print(f"Candidate pairs: {len(candidates):,}")
    print(f"Duplicate pairs: {duplicate_count:,}")
    print(
        f"Unique source1 records: "
        f"{candidates['source1_id'].nunique():,}"
    )
    print(
        f"Candidate sources: "
        f"{candidates['candidate_source'].nunique():,}"
    )

    print("\nBlocking strategy distribution:")
    print(candidates["blocking_strategy"].value_counts())


# ============================================================
# GROUND TRUTH → PAIRS
# ============================================================

def build_ground_truth_pairs(
    ground_truth: pd.DataFrame,
) -> set[tuple[str, str]]:
    """
    Convert train_ground_truth.tsv into true entity pairs.

    Example:

    S1-123 -> S2-456,S2-789,S3-111

    becomes:

    (S1-123, S2-456)
    (S1-123, S2-789)
    (S1-123, S3-111)
    """

    required_columns = {
        "source1_entity_id",
        "matched_entity_ids",
    }

    missing = required_columns - set(ground_truth.columns)

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
                    (
                        source1_id,
                        matched_id,
                    )
                )

    return true_pairs


# ============================================================
# CANDIDATES → PAIRS
# ============================================================

def build_candidate_pairs(
    candidates: pd.DataFrame,
) -> set[tuple[str, str]]:
    """Convert candidate dataframe into unique entity pairs."""

    required_columns = {
        "source1_id",
        "candidate_id",
    }

    missing = required_columns - set(candidates.columns)

    if missing:
        raise ValueError(
            f"Missing candidate columns: {sorted(missing)}"
        )

    candidate_pairs: set[tuple[str, str]] = set()

    for _, row in candidates.iterrows():

        source1_id = str(
            row["source1_id"]
        ).strip()

        candidate_id = str(
            row["candidate_id"]
        ).strip()

        if source1_id and candidate_id:
            candidate_pairs.add(
                (
                    source1_id,
                    candidate_id,
                )
            )

    return candidate_pairs


# ============================================================
# CANDIDATE RECALL
# ============================================================

def calculate_candidate_recall(
    candidates: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> dict[str, Any]:
    """
    Calculate candidate recall.

    Recall =
        true matches recovered by blocking
        ---------------------------------
        total true matches
    """

    candidate_pairs = build_candidate_pairs(
        candidates
    )

    true_pairs = build_ground_truth_pairs(
        ground_truth
    )

    recovered_pairs = (
        candidate_pairs & true_pairs
    )

    missed_pairs = (
        true_pairs - candidate_pairs
    )

    total_true_matches = len(true_pairs)
    recovered_matches = len(recovered_pairs)
    missed_matches = len(missed_pairs)

    if total_true_matches > 0:
        recall = (
            recovered_matches
            / total_true_matches
        )
    else:
        recall = 0.0

    return {
        "total_true_matches": total_true_matches,
        "recovered_matches": recovered_matches,
        "missed_matches": missed_matches,
        "candidate_recall": recall,
    }


def print_recall_results(
    results: dict[str, Any],
) -> None:
    """Print candidate recall results."""

    print()
    print("=" * 70)
    print("CANDIDATE RECALL")
    print("=" * 70)

    print(
        f"Total true matches: "
        f"{results['total_true_matches']:,}"
    )

    print(
        f"Recovered matches:  "
        f"{results['recovered_matches']:,}"
    )

    print(
        f"Missed matches:     "
        f"{results['missed_matches']:,}"
    )

    print(
        f"Candidate Recall:   "
        f"{results['candidate_recall']:.4%}"
    )


# ============================================================
# CANDIDATE SIZE STATISTICS
# ============================================================

def calculate_candidate_statistics(
    candidates: pd.DataFrame,
) -> dict[str, Any]:
    """Calculate candidate volume statistics."""

    if candidates.empty:
        return {
            "total_candidates": 0,
            "unique_source1": 0,
            "average_candidates_per_source1": 0.0,
            "median_candidates_per_source1": 0.0,
            "maximum_candidates_per_source1": 0,
        }

    counts = (
        candidates
        .groupby("source1_id")
        .size()
    )

    return {
        "total_candidates": len(candidates),
        "unique_source1": candidates[
            "source1_id"
        ].nunique(),
        "average_candidates_per_source1": counts.mean(),
        "median_candidates_per_source1": counts.median(),
        "maximum_candidates_per_source1": counts.max(),
    }


def print_candidate_statistics(
    statistics: dict[str, Any],
) -> None:
    """Print candidate volume statistics."""

    print()
    print("=" * 70)
    print("CANDIDATE SIZE STATISTICS")
    print("=" * 70)

    print(
        f"Total candidates: "
        f"{statistics['total_candidates']:,}"
    )

    print(
        f"Source-1 entities with candidates: "
        f"{statistics['unique_source1']:,}"
    )

    print(
        f"Average candidates / Source-1: "
        f"{statistics['average_candidates_per_source1']:.2f}"
    )

    print(
        f"Median candidates / Source-1: "
        f"{statistics['median_candidates_per_source1']:.2f}"
    )

    print(
        f"Maximum candidates / Source-1: "
        f"{statistics['maximum_candidates_per_source1']:,}"
    )


# ============================================================
# ZERO-CANDIDATE ANALYSIS
# ============================================================

def calculate_zero_candidate_entities(
    source1: pd.DataFrame,
    candidates: pd.DataFrame,
) -> dict[str, Any]:
    """
    Find Source-1 records for which M2 generated no candidates.
    """

    source1_ids = set(
        source1["entity_id"].astype(str)
    )

    candidate_source1_ids = set(
        candidates["source1_id"].astype(str)
    )

    zero_candidate_ids = (
        source1_ids - candidate_source1_ids
    )

    total_source1 = len(source1_ids)
    zero_count = len(zero_candidate_ids)

    zero_rate = (
        zero_count / total_source1
        if total_source1
        else 0.0
    )

    return {
        "total_source1": total_source1,
        "zero_candidate_entities": zero_count,
        "zero_candidate_rate": zero_rate,
        "zero_candidate_ids": zero_candidate_ids,
    }


def print_zero_candidate_results(
    results: dict[str, Any],
) -> None:
    """Print zero-candidate statistics."""

    print()
    print("=" * 70)
    print("ZERO-CANDIDATE ANALYSIS")
    print("=" * 70)

    print(
        f"Total Source-1 entities: "
        f"{results['total_source1']:,}"
    )

    print(
        f"Zero-candidate entities: "
        f"{results['zero_candidate_entities']:,}"
    )

    print(
        f"Zero-candidate rate: "
        f"{results['zero_candidate_rate']:.4%}"
    )


# ============================================================
# COMPLETE M2 EVALUATION
# ============================================================

def evaluate_blocking(
    source1: pd.DataFrame,
    candidates: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> dict[str, Any]:
    """
    Run the complete M2 evaluation.

    Evaluates:

    1. Candidate schema
    2. Candidate recall
    3. Candidate volume
    4. Zero-candidate records
    """

    validate_candidates(candidates)

    recall_results = calculate_candidate_recall(
        candidates,
        ground_truth,
    )

    candidate_statistics = calculate_candidate_statistics(
        candidates
    )

    zero_candidate_results = (
        calculate_zero_candidate_entities(
            source1,
            candidates,
        )
    )

    print_recall_results(
        recall_results
    )

    print_candidate_statistics(
        candidate_statistics
    )

    print_zero_candidate_results(
        zero_candidate_results
    )

    return {
        "recall": recall_results,
        "candidate_statistics": candidate_statistics,
        "zero_candidate_analysis": zero_candidate_results,
    }


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":
    print(
        "M2 validation module loaded successfully."
    )

