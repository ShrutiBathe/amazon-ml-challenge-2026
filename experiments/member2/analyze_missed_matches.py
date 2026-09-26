
import pandas as pd
from sympy import python


# ============================================================
# LOAD DATA
# ============================================================

print("Loading Source 1...")

s1 = pd.read_csv(
    "dataset/processed/train/train_source1.tsv",
    sep="\t",
    nrows=10000
)

print("Loading Source 2...")

s2 = pd.read_csv(
    "dataset/processed/train/train_source2.tsv",
    sep="\t"
)

print("Loading Source 3...")

s3 = pd.read_csv(
    "dataset/processed/train/train_source3.tsv",
    sep="\t"
)

print("Loading ground truth...")

gt = pd.read_csv(
    "dataset/train/train_ground_truth.tsv",
    sep="\t"
)


# ============================================================
# FILTER GROUND TRUTH TO OUR S1 SAMPLE
# ============================================================

gt = gt[
    gt["source1_entity_id"].isin(
        s1["entity_id"]
    )
]


# ============================================================
# KEEP ONLY REQUIRED COLUMNS
# ============================================================

print("Preparing lookup tables...")

cols = [
    "entity_id",
    "name_prefix_6",
    "name_first_token",
    "address_first_token",
    "country_normalized",
    "name_no_suffix_compact",
    "name_compact",
    "address_compact",
    "address_normalized",
]


s1 = s1[
    cols +
    [
        "business_name",
        "business_address"
    ]
]

s2 = s2[
    cols +
    [
        "business_name",
        "business_address"
    ]
]

s3 = s3[
    cols +
    [
        "business_name",
        "business_address"
    ]
]


# ============================================================
# FAST DICTIONARY LOOKUPS
# ============================================================

s1_records = (
    s1
    .set_index("entity_id")
    .to_dict("index")
)

s2_records = (
    s2
    .set_index("entity_id")
    .to_dict("index")
)

s3_records = (
    s3
    .set_index("entity_id")
    .to_dict("index")
)


# ============================================================
# ANALYSIS VARIABLES
# ============================================================

print("Analyzing true matches...")

total = 0

v41_covered = 0

missed = 0


stats = {

    "country": 0,

    "name_no_suffix": 0,

    "name_compact": 0,

    "address_compact": 0,

    "address_normalized": 0,

    "shared_address_1plus": 0,

    "shared_address_2plus": 0,

    "shared_address_3plus": 0,

    "shared_address_4plus": 0,

}


examples = []


# ============================================================
# PROCESS GROUND TRUTH
# ============================================================

for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]

    a = s1_records.get(s1_id)

    if a is None:
        continue


    matched_ids = str(
        row["matched_entity_ids"]
    ).split(",")


    for candidate_id in matched_ids:

        candidate_id = candidate_id.strip()


        # ----------------------------------------------------
        # FIND CANDIDATE
        # ----------------------------------------------------

        if candidate_id.startswith("S2-"):

            b = s2_records.get(candidate_id)

        elif candidate_id.startswith("S3-"):

            b = s3_records.get(candidate_id)

        else:

            continue


        if b is None:
            continue


        total += 1


        # ====================================================
        # V3 KEYS
        # ====================================================

        name_prefix = (
            a["name_prefix_6"]
            == b["name_prefix_6"]
        )


        first_token = (
            a["name_first_token"]
            == b["name_first_token"]
        )


        address_first_token = (
            a["address_first_token"]
            == b["address_first_token"]
        )


        # ====================================================
        # V4 ADDRESS TOKEN OVERLAP
        # ====================================================

        a_address = str(
            a["address_normalized"]
        )

        b_address = str(
            b["address_normalized"]
        )


        a_tokens = set(
            a_address.split()
        )

        b_tokens = set(
            b_address.split()
        )


        shared_address_tokens = len(
            a_tokens & b_tokens
        )


        # ====================================================
        # V4.1 BLOCKING RULE
        #
        # country + 2 or more shared address tokens
        # ====================================================

        country = (
            a["country_normalized"]
            == b["country_normalized"]
        )


        country_address_overlap = (
            country
            and
            shared_address_tokens >= 2
        )


        # ====================================================
        # CHECK V4.1 COVERAGE
        # ====================================================

        if (
            name_prefix
            or first_token
            or address_first_token
            or country_address_overlap
        ):

            v41_covered += 1

            continue


        # ====================================================
        # MISSED BY V4.1
        # ====================================================

        missed += 1


        # ====================================================
        # ALTERNATIVE KEYS
        # ====================================================

        name_no_suffix = (
            a["name_no_suffix_compact"]
            == b["name_no_suffix_compact"]
        )


        name_compact = (
            a["name_compact"]
            == b["name_compact"]
        )


        address_compact = (
            a["address_compact"]
            == b["address_compact"]
        )


        address_normalized = (
            a["address_normalized"]
            == b["address_normalized"]
        )


        # ====================================================
        # UPDATE STATISTICS
        # ====================================================

        if country:

            stats["country"] += 1


        if name_no_suffix:

            stats["name_no_suffix"] += 1


        if name_compact:

            stats["name_compact"] += 1


        if address_compact:

            stats["address_compact"] += 1


        if address_normalized:

            stats["address_normalized"] += 1


        if shared_address_tokens >= 1:

            stats["shared_address_1plus"] += 1


        if shared_address_tokens >= 2:

            stats["shared_address_2plus"] += 1


        if shared_address_tokens >= 3:

            stats["shared_address_3plus"] += 1


        if shared_address_tokens >= 4:

            stats["shared_address_4plus"] += 1


        # ====================================================
        # SAVE EXAMPLES
        # ====================================================

        if len(examples) < 20:

            examples.append({

                "source1_id":
                    s1_id,

                "candidate_id":
                    candidate_id,

                "source1_name":
                    a["business_name"],

                "candidate_name":
                    b["business_name"],

                "source1_address":
                    a["business_address"],

                "candidate_address":
                    b["business_address"],

                "country":
                    country,

                "shared_address_tokens":
                    shared_address_tokens,

                "name_no_suffix":
                    name_no_suffix,

                "name_compact":
                    name_compact,

                "address_compact":
                    address_compact,

                "address_normalized":
                    address_normalized,

            })


# ============================================================
# FINAL SUMMARY
# ============================================================

print()

print("=" * 70)

print(
    "V4.1 MISSED-MATCH ANALYSIS"
)

print("=" * 70)


print(
    f"Total true matches:       {total:,}"
)

print(
    f"Covered by V4.1 keys:     {v41_covered:,}"
)

print(
    f"Missed by V4.1 keys:      {missed:,}"
)


if total:

    coverage = (
        v41_covered
        / total
        * 100
    )

    missed_rate = (
        missed
        / total
        * 100
    )

    print(
        f"V4.1 coverage:           "
        f"{coverage:.2f}%"
    )

    print(
        f"Missed rate:              "
        f"{missed_rate:.2f}%"
    )


# ============================================================
# ALTERNATIVE KEY ANALYSIS
# ============================================================

print()

print("=" * 70)

print(
    "ALTERNATIVE KEYS AMONG V4.1 MISSED MATCHES"
)

print("=" * 70)


if missed:

    for key, value in stats.items():

        pct = (
            value
            / missed
            * 100
        )

        print(
            f"{key:30s}"
            f"{value:8,}"
            f" ({pct:6.2f}%)"
        )


# ============================================================
# MISSED MATCH EXAMPLES
# ============================================================

print()

print("=" * 70)

print(
    "EXAMPLE MISSED MATCHES"
)

print("=" * 70)


for x in examples:

    print()

    print(
        "S1:",
        x["source1_id"]
    )

    print(
        "Candidate:",
        x["candidate_id"]
    )

    print(
        "S1 Name:",
        x["source1_name"]
    )

    print(
        "Candidate Name:",
        x["candidate_name"]
    )

    print(
        "S1 Address:",
        x["source1_address"]
    )

    print(
        "Candidate Address:",
        x["candidate_address"]
    )

    print(
        "Country:",
        x["country"],
        "| Shared Address Tokens:",
        x["shared_address_tokens"],
        "| NoSuffix:",
        x["name_no_suffix"],
        "| Compact:",
        x["name_compact"],
        "| AddressCompact:",
        x["address_compact"]
    )


# ============================================================
# END
# ============================================================

print()

print("=" * 70)

print(
    "ANALYSIS COMPLETE"
)

print("=" * 70)
