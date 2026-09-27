from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.matching.model import (
    load_matcher,
    get_feature_columns,
    predict_match_probabilities,
)


BASE = Path(__file__).resolve().parents[1]

FEATURES_PATH = (
    BASE
    / "dataset"
    / "processed"
    / "m3_test"
    / "m3_test_features.tsv"
)

MODEL_PATH = (
    BASE
    / "dataset"
    / "processed"
    / "m3_test"
    / "m3_matcher.joblib"
)

OUTPUT_PATH = (
    BASE
    / "dataset"
    / "processed"
    / "m3_test"
    / "m3_decision_results.tsv"
)


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

PROBABILITY_THRESHOLD = 0.50
MARGIN_THRESHOLD = 0.10


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

print("=" * 70)
print("M3 EVIDENCE-AWARE DECISION TEST")
print("=" * 70)

print("\nLoading features...")

features = pd.read_csv(
    FEATURES_PATH,
    sep="\t",
)

print(
    f"Feature rows: {len(features):,}"
)

print("\nLoading trained matcher...")

model, feature_columns = load_matcher(
    MODEL_PATH
)

print(
    f"ML features: {len(feature_columns)}"
)


# ------------------------------------------------------------
# PREDICT
# ------------------------------------------------------------

print("\nGenerating probabilities...")

predictions = predict_match_probabilities(
    model,
    features,
    feature_columns,
)

print(
    f"Predictions: {len(predictions):,}"
)


# ------------------------------------------------------------
# ADD EVIDENCE
# ------------------------------------------------------------

evidence_columns = [
    "name_exact",
    "name_compact_exact",
    "name_no_suffix_exact",
    "name_no_suffix_compact_exact",
    "name_dedup_exact",
    "name_jaccard",
    "name_compact_jaccard",
    "name_no_suffix_jaccard",
    "name_containment",
    "name_char_similarity",
    "address_exact",
    "address_compact_exact",
    "address_jaccard",
    "address_containment",
    "address_char_similarity",
    "address_numeric_overlap",
    "address_numeric_conflict",
    "country_match",
    "country_conflict",
    "name_support_count",
    "address_support_count",
    "total_support_count",
    "contradiction_count",
    "strong_name_weak_address",
    "strong_name_numeric_conflict",
    "country_conflict_with_name_match",
]

available_evidence = [
    c
    for c in evidence_columns
    if c in features.columns
]

results = predictions.copy()

for column in available_evidence:
    results[column] = features[column].values


# ------------------------------------------------------------
# EVIDENCE SCORE
# ------------------------------------------------------------

results["evidence_score"] = (
    results["name_support_count"].fillna(0)
    + results["address_support_count"].fillna(0)
    + results["total_support_count"].fillna(0)
)


# ------------------------------------------------------------
# CONTRADICTION PENALTY
# ------------------------------------------------------------

results["contradiction_penalty"] = (
    results["contradiction_count"].fillna(0)
    + results["address_numeric_conflict"].fillna(0)
    + results["strong_name_numeric_conflict"].fillna(0)
    + results["country_conflict_with_name_match"].fillna(0)
)


# ------------------------------------------------------------
# SORT BY SOURCE1 + PROBABILITY
# ------------------------------------------------------------

results = results.sort_values(
    [
        "source1_id",
        "match_probability",
    ],
    ascending=[
        True,
        False,
    ],
)


# ------------------------------------------------------------
# CANDIDATE MARGIN
# ------------------------------------------------------------

results["second_best_probability"] = (
    results
    .groupby("source1_id")["match_probability"]
    .shift(-1)
)

results["second_best_probability"] = (
    results["second_best_probability"]
    .fillna(0.0)
)

results["probability_margin"] = (
    results["match_probability"]
    - results["second_best_probability"]
)


# ------------------------------------------------------------
# DECISION
# ------------------------------------------------------------

results["decision"] = "NO_MATCH"


strong_match = (
    (results["match_probability"] >= PROBABILITY_THRESHOLD)
    &
    (results["contradiction_penalty"] == 0)
)

results.loc[
    strong_match,
    "decision"
] = "MATCH"


# ------------------------------------------------------------
# IMPORTANT:
# Preserve M4-compatible core columns
# ------------------------------------------------------------

output_columns = [
    "source1_id",
    "candidate_id",
    "candidate_source",
    "match_probability",
    "evidence_score",
    "contradiction_penalty",
    "probability_margin",
    "decision",
]

output = results[
    output_columns
].copy()


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

output.to_csv(
    OUTPUT_PATH,
    sep="\t",
    index=False,
)


# ------------------------------------------------------------
# REPORT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DECISION RESULTS")
print("=" * 70)

print(
    f"\nTotal candidates : {len(output):,}"
)

print(
    f"MATCH decisions  : "
    f"{(output['decision'] == 'MATCH').sum():,}"
)

print(
    f"NO_MATCH         : "
    f"{(output['decision'] == 'NO_MATCH').sum():,}"
)

print(
    f"\nProbability >= "
    f"{PROBABILITY_THRESHOLD}: "
    f"{(output['match_probability'] >= PROBABILITY_THRESHOLD).sum():,}"
)

print(
    "\nTop probability examples:"
)

print(
    output[
        [
            "source1_id",
            "candidate_id",
            "match_probability",
            "evidence_score",
            "contradiction_penalty",
            "probability_margin",
            "decision",
        ]
    ]
    .sort_values(
        "match_probability",
        ascending=False,
    )
    .head(20)
    .to_string(index=False)
)

print(
    f"\nOutput saved to:\n{OUTPUT_PATH}"
)

print("\n" + "=" * 70)
print("M3 DECISION TEST COMPLETE")
print("=" * 70)