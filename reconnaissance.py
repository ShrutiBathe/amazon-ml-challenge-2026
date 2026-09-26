import pandas as pd
from pathlib import Path

BASE = Path("dataset")

files = [
    "train/train_source1.tsv",
    "train/train_source2.tsv",
    "train/train_source3.tsv",
    "train/train_ground_truth.tsv",
    "test/test_source1.tsv",
    "test/test_source2.tsv",
    "test/test_source3.tsv",
]

for file in files:
    path = BASE / file

    print("\n" + "=" * 70)
    print(file)
    print("=" * 70)

    if not path.exists():
        print("FILE NOT FOUND:", path)
        continue

    df = pd.read_csv(path, sep="\t")

    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    if "country" in df.columns:
        print("\nCountry distribution:")
        print(df["country"].value_counts(dropna=False))

    if "business_name" in df.columns:
        print("\nUnique business names:", df["business_name"].nunique())

    if "business_address" in df.columns:
        print("Unique addresses:", df["business_address"].nunique())

print("\n" + "=" * 70)
print("RECONNAISSANCE COMPLETE")
print("=" * 70)