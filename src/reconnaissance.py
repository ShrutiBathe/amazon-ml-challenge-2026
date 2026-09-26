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

    df = pd.read_csv(path, sep="\t")

    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    for col in df.columns:
        if col in ["business_name", "business_address", "country"]:
            print(f"\n{col} unique:", df[col].nunique())

    if "country" in df.columns:
        print("\nCountry distribution:")
        print(df["country"].value_counts(dropna=False))

print("\n\nDONE")