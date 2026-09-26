import pandas as pd

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

gt = gt[gt["source1_entity_id"].isin(s1["entity_id"])]

s1i = s1.set_index("entity_id")
s2i = s2.set_index("entity_id")
s3i = s3.set_index("entity_id")

combos = {
    "name_prefix OR first_token": 0,
    "name_prefix OR address_first_token": 0,
    "first_token OR address_first_token": 0,
    "all_three": 0,
    "name_prefix AND country": 0,
    "first_token AND country": 0,
    "address_first_token AND country": 0,
}

total = 0

print("Calculating key combinations...")

for _, row in gt.iterrows():

    source1_id = row["source1_entity_id"]

    if source1_id not in s1i.index:
        continue

    a = s1i.loc[source1_id]

    for candidate_id in str(row["matched_entity_ids"]).split(","):

        if candidate_id.startswith("S2-"):
            source = s2i
        elif candidate_id.startswith("S3-"):
            source = s3i
        else:
            continue

        if candidate_id not in source.index:
            continue

        b = source.loc[candidate_id]

        total += 1

        name_prefix = (
            a["name_prefix_6"] == b["name_prefix_6"]
        )

        first_token = (
            a["name_first_token"] == b["name_first_token"]
        )

        address_first_token = (
            a["address_first_token"] == b["address_first_token"]
        )

        country = (
            a["country_normalized"] == b["country_normalized"]
        )

        if name_prefix or first_token:
            combos["name_prefix OR first_token"] += 1

        if name_prefix or address_first_token:
            combos["name_prefix OR address_first_token"] += 1

        if first_token or address_first_token:
            combos["first_token OR address_first_token"] += 1

        if name_prefix or first_token or address_first_token:
            combos["all_three"] += 1

        if name_prefix and country:
            combos["name_prefix AND country"] += 1

        if first_token and country:
            combos["first_token AND country"] += 1

        if address_first_token and country:
            combos["address_first_token AND country"] += 1


print()
print("=" * 70)
print("BLOCKING KEY COMBINATION ANALYSIS")
print("=" * 70)

print(f"Total true matches: {total:,}")
print()

for key, count in combos.items():

    percentage = (
        count / total * 100
        if total
        else 0
    )

    print(
        f"{key:<40} "
        f"{count:>8,} "
        f"({percentage:>6.2f}%)"
    )

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)