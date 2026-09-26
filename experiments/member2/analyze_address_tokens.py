import pandas as pd
from collections import Counter

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

gt = gt[
    gt["source1_entity_id"].isin(s1["entity_id"])
]

s1i = s1.set_index("entity_id")
s2i = s2.set_index("entity_id")
s3i = s3.set_index("entity_id")

# Tokens that are too common to be useful by themselves
STOP_TOKENS = {
    "no",
    "nr",
    "near",
    "road",
    "rd",
    "street",
    "st",
    "avenue",
    "ave",
    "lane",
    "ln",
    "floor",
    "fl",
    "flat",
    "unit",
    "house",
    "h",
    "block",
    "plot",
    "shop",
    "office",
    "building",
    "bldg",
    "india",
    "us",
}


def get_tokens(value):
    if pd.isna(value):
        return []

    text = str(value).lower()

    tokens = []

    for token in text.replace(",", " ").replace(".", " ").split():

        token = token.strip()

        if not token:
            continue

        if token in STOP_TOKENS:
            continue

        if len(token) < 3:
            continue

        tokens.append(token)

    return tokens


def overlap(a, b):
    a_tokens = set(get_tokens(a))
    b_tokens = set(get_tokens(b))

    if not a_tokens or not b_tokens:
        return 0

    return len(a_tokens & b_tokens)


stats = Counter()

total = 0

examples = []

print("Analyzing address token overlap...")

for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]

    if s1_id not in s1i.index:
        continue

    a = s1i.loc[s1_id]

    for candidate_id in str(
        row["matched_entity_ids"]
    ).split(","):

        candidate_id = candidate_id.strip()

        if candidate_id.startswith("S2-"):
            b = s2i.loc[candidate_id] if candidate_id in s2i.index else None
        elif candidate_id.startswith("S3-"):
            b = s3i.loc[candidate_id] if candidate_id in s3i.index else None
        else:
            continue

        if b is None:
            continue

        total += 1

        ov = overlap(
            a["address_normalized"],
            b["address_normalized"]
        )

        if ov >= 1:
            stats["1+ shared address tokens"] += 1

        if ov >= 2:
            stats["2+ shared address tokens"] += 1

        if ov >= 3:
            stats["3+ shared address tokens"] += 1

        if ov >= 4:
            stats["4+ shared address tokens"] += 1

        if ov == 0:
            stats["zero shared tokens"] += 1

        if len(examples) < 20 and ov >= 2:
            examples.append(
                (
                    s1_id,
                    candidate_id,
                    a["business_address"],
                    b["business_address"],
                    ov,
                )
            )


print()
print("=" * 70)
print("ADDRESS TOKEN OVERLAP")
print("=" * 70)

print("Total true matches:", f"{total:,}")

for key in [
    "1+ shared address tokens",
    "2+ shared address tokens",
    "3+ shared address tokens",
    "4+ shared address tokens",
    "zero shared tokens",
]:

    count = stats[key]

    pct = (
        count / total * 100
        if total
        else 0
    )

    print(
        f"{key:30s}"
        f"{count:8,}"
        f" ({pct:6.2f}%)"
    )


print()
print("=" * 70)
print("EXAMPLES WITH 2+ SHARED ADDRESS TOKENS")
print("=" * 70)

for x in examples:

    print()
    print("S1:", x[0])
    print("Candidate:", x[1])
    print("Shared tokens:", x[4])
    print("S1:", x[2])
    print("Candidate:", x[3])