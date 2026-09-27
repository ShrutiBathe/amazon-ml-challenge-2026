from pathlib import Path
import sys
import pandas as pd
import joblib

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from src.matching.features import build_pair_features

MODEL = BASE / "dataset/processed/m3_test/m3_matcher_final.joblib"
S1 = BASE / "dataset/processed/test/test_source1.tsv"
S2 = BASE / "dataset/processed/test/test_source2.tsv"
S3 = BASE / "dataset/processed/test/test_source3.tsv"

CANDIDATES = [
    BASE / "dataset/processed/m3_candidates/test_source2_exact_name.tsv",
    BASE / "dataset/processed/m3_candidates/test_source3_exact_name.tsv",
]

OUT = BASE / "dataset/processed/m3_test/test_m3_predictions.tsv"
CHUNK = 25000

print("=" * 70)
print("M3 FINAL TEST INFERENCE")
print("=" * 70)

# Load model
payload = joblib.load(MODEL)
model = payload["model"]
feature_columns = payload["feature_columns"]

print(f"Model loaded: {MODEL.name}")
print(f"Features: {len(feature_columns)}")

# Source1 is shared by both candidate sets.
print("\nLoading Source1...")
source1 = pd.read_csv(S1, sep="\t", dtype=str)
print(f"Source1 rows: {len(source1):,}")

# Required columns are determined from the feature builder/model.
usecols = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
    "name_normalized",
    "name_compact",
    "name_no_suffix",
    "name_no_suffix_compact",
    "name_dedup",
    "address_normalized",
    "address_compact",
    "country_normalized",
    "name_prefix",
    "name_prefix_6",
    "name_first_token",
    "address_first_token",
]

print("Loading Source2...")
source2 = pd.read_csv(S2, sep="\t", dtype=str, usecols=usecols)
print(f"Source2 rows: {len(source2):,}")

print("Loading Source3...")
source3 = pd.read_csv(S3, sep="\t", dtype=str, usecols=usecols)
print(f"Source3 rows: {len(source3):,}")

candidate_sources = {
    "source2": source2,
    "source3": source3,
}

if OUT.exists():
    OUT.unlink()

first_write = True
total = 0

for candidate_file in CANDIDATES:

    print("\n" + "=" * 70)
    print(f"Processing: {candidate_file.name}")
    print("=" * 70)

    for candidates in pd.read_csv(
        candidate_file,
        sep="\t",
        dtype=str,
        chunksize=CHUNK,
    ):

        features = build_pair_features(
            source1,
            candidate_sources,
            candidates,
        )

        X = features[feature_columns]

        probabilities = model.predict_proba(X)[:, 1]

        result = candidates[
            [
                "source1_id",
                "candidate_id",
                "candidate_source",
                "blocking_strategy",
            ]
        ].copy()

        result["match_probability"] = probabilities

        result.to_csv(
            OUT,
            sep="\t",
            index=False,
            mode="w" if first_write else "a",
            header=first_write,
        )

        first_write = False
        total += len(result)

        print(
            f"\rProcessed candidates: {total:,}",
            end="",
            flush=True,
        )

print()
print("\n" + "=" * 70)
print(f"TOTAL M3 PREDICTIONS: {total:,}")
print(f"OUTPUT: {OUT}")
print("=" * 70)
