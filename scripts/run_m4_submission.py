from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]

PREDICTIONS = (
    BASE / "dataset" / "processed" / "m3_test" / "test_m3_predictions.tsv"
)

TEST_SOURCE1 = BASE / "dataset" / "test" / "test_source1.tsv"

OUTPUT_DIR = BASE / "dataset" / "processed" / "m4_final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUTPUT_DIR / "matching_results.tsv"

THRESHOLD = 0.90


def main():
    print("=" * 70)
    print("M4 FINAL SUBMISSION GENERATION")
    print("=" * 70)

    print("\nLoading test Source1...")
    source1 = pd.read_csv(
        TEST_SOURCE1,
        sep="\t",
        usecols=["entity_id"],
    )

    print(f"Test Source1 entities: {len(source1):,}")

    print("\nLoading M3 predictions...")
    predictions = pd.read_csv(
        PREDICTIONS,
        sep="\t",
        usecols=[
            "source1_id",
            "candidate_id",
            "match_probability",
        ],
    )

    print(f"Predictions: {len(predictions):,}")

    # Keep high-confidence candidate matches.
    matches = predictions[
        predictions["match_probability"] >= THRESHOLD
    ].copy()

    print(f"Candidates >= {THRESHOLD:.2f}: {len(matches):,}")

    # Sort strongest candidates first.
    matches = matches.sort_values(
        ["source1_id", "match_probability"],
        ascending=[True, False],
    )

    # Keep all qualifying matches for each Source1.
    grouped = (
        matches
        .groupby("source1_id")["candidate_id"]
        .apply(lambda x: ",".join(dict.fromkeys(x)))
        .reset_index()
    )

    grouped.columns = [
        "source1_entity_id",
        "matched_entity_ids",
    ]

    # Include Source1 entities that have no accepted match.
    result = source1.rename(
        columns={"entity_id": "source1_entity_id"}
    ).merge(
        grouped,
        on="source1_entity_id",
        how="left",
    )

    result["matched_entity_ids"] = (
        result["matched_entity_ids"]
        .fillna("")
    )

    result.to_csv(
        OUTPUT,
        sep="\t",
        index=False,
    )

    matched_entities = (
        result["matched_entity_ids"].ne("").sum()
    )

    no_match_entities = (
        result["matched_entity_ids"].eq("").sum()
    )

    print("\n" + "=" * 70)
    print("FINAL SUBMISSION SUMMARY")
    print("=" * 70)

    print(f"Output rows: {len(result):,}")
    print(f"Entities with matches: {matched_entities:,}")
    print(f"Entities with no match: {no_match_entities:,}")
    print(
        f"Match coverage: "
        f"{matched_entities / len(result):.4%}"
    )

    print(f"\nOutput: {OUTPUT}")

    print("\nSample:")
    print(result.head(10).to_string(index=False))

    print("\nM4 submission generation completed.")


if __name__ == "__main__":
    main()