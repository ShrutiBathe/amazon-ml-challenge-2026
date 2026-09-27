from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]

PREDICTIONS = (
    BASE / "dataset" / "processed" / "m3_test" / "test_m3_predictions.tsv"
)

OUTPUT_DIR = (
    BASE / "dataset" / "processed" / "m4_final"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MATCHING_RESULTS = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS = OUTPUT_DIR / "candidate_pairs.tsv"

# Precision-focused final decision settings
PROBABILITY_THRESHOLD = 0.90
MARGIN_THRESHOLD = 0.10

CHUNK_SIZE = 500_000


def main():
    print("=" * 70)
    print("M4 FINAL ENTITY RESOLUTION")
    print("=" * 70)

    print(f"\nInput: {PREDICTIONS}")
    print(f"Output directory: {OUTPUT_DIR}")

    # ---------------------------------------------------------
    # PASS 1: Load predictions
    # ---------------------------------------------------------

    print("\nLoading M3 predictions...")

    df = pd.read_csv(PREDICTIONS, sep="\t")

    print(f"Total candidate predictions: {len(df):,}")
    print(f"Source1 entities: {df['source1_id'].nunique():,}")

    # ---------------------------------------------------------
    # Rank candidates within each Source1 entity
    # ---------------------------------------------------------

    print("\nRanking candidates...")

    df = df.sort_values(
        ["source1_id", "match_probability"],
        ascending=[True, False],
        kind="mergesort",
    )

    df["rank"] = df.groupby("source1_id").cumcount() + 1

    df["second_best_probability"] = (
        df.groupby("source1_id")["match_probability"]
        .shift(-1)
        .fillna(0.0)
    )

    df["probability_margin"] = (
        df["match_probability"] - df["second_best_probability"]
    )

    # ---------------------------------------------------------
    # Candidate-pair output
    # ---------------------------------------------------------

    candidate_pairs = df[
        [
            "source1_id",
            "candidate_id",
            "candidate_source",
            "blocking_strategy",
            "match_probability",
        ]
    ].copy()

    candidate_pairs.to_csv(
        CANDIDATE_PAIRS,
        sep="\t",
        index=False,
    )

    print(f"\nCandidate pairs written: {len(candidate_pairs):,}")
    print(f"File: {CANDIDATE_PAIRS}")

    # ---------------------------------------------------------
    # Final entity-level decision
    # ---------------------------------------------------------

    best = df[df["rank"] == 1].copy()

    best["decision"] = "NO_MATCH"

    strong_match = (
        (best["match_probability"] >= PROBABILITY_THRESHOLD)
        & (best["probability_margin"] >= MARGIN_THRESHOLD)
    )

    best.loc[strong_match, "decision"] = "MATCH"

    # ---------------------------------------------------------
    # Final output
    # ---------------------------------------------------------

    matching_results = best[
        [
            "source1_id",
            "candidate_id",
            "candidate_source",
            "match_probability",
            "probability_margin",
            "decision",
        ]
    ].copy()

    matching_results.to_csv(
        MATCHING_RESULTS,
        sep="\t",
        index=False,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL DECISION SUMMARY")
    print("=" * 70)

    print(
        f"Source1 entities: {len(matching_results):,}"
    )

    print(
        f"MATCH: "
        f"{(matching_results['decision'] == 'MATCH').sum():,}"
    )

    print(
        f"NO_MATCH: "
        f"{(matching_results['decision'] == 'NO_MATCH').sum():,}"
    )

    print(
        f"Match rate: "
        f"{(matching_results['decision'] == 'MATCH').mean():.4%}"
    )

    print(f"\nMatching results: {MATCHING_RESULTS}")
    print(f"Candidate pairs: {CANDIDATE_PAIRS}")

    print("\nM4 final decision completed.")


if __name__ == "__main__":
    main()