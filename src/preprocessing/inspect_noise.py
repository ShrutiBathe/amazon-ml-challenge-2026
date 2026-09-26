import pandas as pd
from pathlib import Path

BASE = Path("dataset")

files = {
    "S1": BASE / "train/train_source1.tsv",
    "S2": BASE / "train/train_source2.tsv",
    "S3": BASE / "train/train_source3.tsv",
}

for source, path in files.items():

    print("\n" + "=" * 80)
    print(f"{source} SAMPLE")
    print("=" * 80)

    df = pd.read_csv(
        path,
        sep="\t",
        nrows=20
    )

    print(
        df[
            [
                "entity_id",
                "business_name",
                "business_address",
                "country"
            ]
        ].to_string(index=False)
    )

print("\n" + "=" * 80)
print("GROUND TRUTH SAMPLE")
print("=" * 80)

gt = pd.read_csv(
    BASE / "train/train_ground_truth.tsv",
    sep="\t",
    nrows=30
)

print(gt.to_string(index=False))