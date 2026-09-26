from pathlib import Path
from collections import Counter

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

GT_PATH = (
    BASE_DIR
    / "dataset"
    / "train"
    / "train_ground_truth.tsv"
)

PROCESSED_DIR = (
    BASE_DIR
    / "dataset"
    / "processed"
    / "train"
)

OUTPUT_DIR = (
    BASE_DIR
    / "dataset"
    / "processed"
    / "noise_dna"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_LIMIT = 500_000


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth():

    print("Loading ground truth...")

    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    pairs = []

    for row in gt.itertuples(index=False):

        s1_id = row.source1_entity_id
        matched = row.matched_entity_ids

        if not matched:
            continue

        for entity_id in matched.split(","):

            entity_id = entity_id.strip()

            if not entity_id:
                continue

            if entity_id.startswith("S2-"):
                source = "S2"

            elif entity_id.startswith("S3-"):
                source = "S3"

            else:
                continue

            pairs.append(
                (
                    s1_id,
                    entity_id,
                    source
                )
            )

    pairs_df = pd.DataFrame(
        pairs,
        columns=[
            "source1_entity_id",
            "matched_entity_id",
            "source"
        ]
    )

    print(
        f"Confirmed positive pairs: "
        f"{len(pairs_df):,}"
    )

    return pairs_df


# ============================================================
# LOAD REQUIRED RECORDS
# ============================================================

def load_processed_file(source):

    path = (
        PROCESSED_DIR
        / f"train_source{source}.tsv"
    )

    print(f"Loading Source {source}...")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
            "name_normalized",
            "name_no_suffix",
            "name_dedup",
            "address_normalized",
        ]
    )

    df = df.rename(
        columns={
            "entity_id": "matched_entity_id"
        }
    )

    return df


# ============================================================
# TEXT HELPERS
# ============================================================

def token_set(text):

    if not text:
        return set()

    return set(text.split())


def jaccard(a, b):

    a = token_set(a)
    b = token_set(b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def containment(a, b):

    a = token_set(a)
    b = token_set(b)

    if not a or not b:
        return 0.0

    return len(a & b) / min(
        len(a),
        len(b)
    )


# ============================================================
# ANALYSIS
# ============================================================

def analyze_pairs():

    pairs = load_ground_truth()

    # --------------------------------------------------------
    # Sample positives
    # --------------------------------------------------------

    if len(pairs) > SAMPLE_LIMIT:

        pairs = pairs.sample(
            n=SAMPLE_LIMIT,
            random_state=42
        )

    print(
        f"Analyzing {len(pairs):,} "
        f"positive pairs..."
    )

    # --------------------------------------------------------
    # Load S1
    # --------------------------------------------------------

    s1_path = (
        PROCESSED_DIR
        / "train_source1.tsv"
    )

    s1 = pd.read_csv(
        s1_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "name_normalized",
            "name_no_suffix",
            "name_dedup",
            "address_normalized",
            "country_normalized",
        ]
    )

    s1 = s1.rename(
        columns={
            "entity_id": "source1_entity_id",
            "name_normalized": "s1_name",
            "name_no_suffix": "s1_name_no_suffix",
            "name_dedup": "s1_name_dedup",
            "address_normalized": "s1_address",
            "country_normalized": "s1_country",
        }
    )

    # --------------------------------------------------------
    # Split S2/S3 pairs
    # --------------------------------------------------------

    statistics = []

    for source in ["S2", "S3"]:

        source_number = source[-1]

        source_pairs = pairs[
            pairs["source"] == source
        ].copy()

        print()
        print("=" * 70)
        print(f"ANALYZING {source}")
        print("=" * 70)

        if source_pairs.empty:

            print("No pairs found.")

            continue

        candidate = load_processed_file(
            source_number
        )

        candidate = candidate.rename(
            columns={
                "business_name": "candidate_name_raw",
                "business_address": "candidate_address_raw",
                "name_normalized": "candidate_name",
                "name_no_suffix": "candidate_name_no_suffix",
                "name_dedup": "candidate_name_dedup",
                "address_normalized": "candidate_address",
                "country": "candidate_country",
            }
        )

        # ----------------------------------------------------
        # Merge confirmed pairs only
        # ----------------------------------------------------

        merged = source_pairs.merge(
            s1,
            on="source1_entity_id",
            how="left"
        )

        merged = merged.merge(
            candidate,
            on="matched_entity_id",
            how="left"
        )

        print(
            f"Joined positive pairs: "
            f"{len(merged):,}"
        )

        # ----------------------------------------------------
        # Feature analysis
        # ----------------------------------------------------

        merged["name_exact"] = (
            merged["s1_name"]
            == merged["candidate_name"]
        )

        merged["name_no_suffix_exact"] = (
            merged["s1_name_no_suffix"]
            == merged["candidate_name_no_suffix"]
        )

        merged["name_dedup_exact"] = (
            merged["s1_name_dedup"]
            == merged["candidate_name_dedup"]
        )

        merged["address_exact"] = (
            merged["s1_address"]
            == merged["candidate_address"]
        )

        merged["country_exact"] = (
            merged["s1_country"]
            == merged["candidate_country"]
        )

        merged["name_jaccard"] = merged.apply(
            lambda r: jaccard(
                r["s1_name"],
                r["candidate_name"]
            ),
            axis=1
        )

        merged["name_containment"] = merged.apply(
            lambda r: containment(
                r["s1_name"],
                r["candidate_name"]
            ),
            axis=1
        )

        merged["address_jaccard"] = merged.apply(
            lambda r: jaccard(
                r["s1_address"],
                r["candidate_address"]
            ),
            axis=1
        )

        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        print()
        print("NAME")

        print(
            "Exact normalized:",
            f"{merged['name_exact'].mean() * 100:.2f}%"
        )

        print(
            "Exact after suffix removal:",
            f"{merged['name_no_suffix_exact'].mean() * 100:.2f}%"
        )

        print(
            "Exact after duplicate-token removal:",
            f"{merged['name_dedup_exact'].mean() * 100:.2f}%"
        )

        print()
        print("ADDRESS")

        print(
            "Exact normalized:",
            f"{merged['address_exact'].mean() * 100:.2f}%"
        )

        print()
        print("COUNTRY")

        print(
            "Exact country:",
            f"{merged['country_exact'].mean() * 100:.2f}%"
        )

        print()
        print("NAME JACCARD")

        print(
            merged["name_jaccard"]
            .describe()
            .to_string()
        )

        print()
        print("ADDRESS JACCARD")

        print(
            merged["address_jaccard"]
            .describe()
            .to_string()
        )

        # ----------------------------------------------------
        # Save sample
        # ----------------------------------------------------

        sample_path = (
            OUTPUT_DIR
            / f"positive_pairs_{source}.tsv"
        )

        merged.head(50_000).to_csv(
            sample_path,
            sep="\t",
            index=False
        )

        print()
        print(
            f"Saved sample: {sample_path}"
        )


if __name__ == "__main__":
    analyze_pairs()