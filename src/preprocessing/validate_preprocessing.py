from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "dataset" / "processed"

EXPECTED_COLUMNS = [
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


FILES = [
    ("train", "train_source1.tsv"),
    ("train", "train_source2.tsv"),
    ("train", "train_source3.tsv"),
    ("test", "test_source1.tsv"),
    ("test", "test_source2.tsv"),
    ("test", "test_source3.tsv"),
]


def validate_file(split, filename):
    path = PROCESSED_DIR / split / filename

    print()
    print("=" * 70)
    print(f"VALIDATING: {split}/{filename}")
    print("=" * 70)

    if not path.exists():
        print("ERROR: File does not exist")
        return False

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=1000,
    )

    # --------------------------------------------------------
    # Column validation
    # --------------------------------------------------------

    missing_columns = [
        col for col in EXPECTED_COLUMNS
        if col not in df.columns
    ]

    extra_columns = [
        col for col in df.columns
        if col not in EXPECTED_COLUMNS
    ]

    if missing_columns:
        print("ERROR: Missing columns:")
        for col in missing_columns:
            print(f"  - {col}")
        return False

    print("✓ All expected columns present")

    if extra_columns:
        print("Extra columns:")
        for col in extra_columns:
            print(f"  - {col}")

    # --------------------------------------------------------
    # Basic checks
    # --------------------------------------------------------

    print(f"✓ Sample rows checked: {len(df):,}")

    for column in EXPECTED_COLUMNS:
        null_count = df[column].isna().sum()

        if null_count > 0:
            print(
                f"WARNING: {column} contains "
                f"{null_count} NaN values"
            )

    # --------------------------------------------------------
    # Country values
    # --------------------------------------------------------

    print(
        "Countries:",
        sorted(
            df["country_normalized"]
            .drop_duplicates()
            .tolist()
        )
    )

    # --------------------------------------------------------
    # Sample transformations
    # --------------------------------------------------------

    print()
    print("Sample transformations:")

    sample_columns = [
        "business_name",
        "name_normalized",
        "name_no_suffix",
        "name_dedup",
        "business_address",
        "address_normalized",
        "country",
        "country_normalized",
    ]

    print(
        df[sample_columns]
        .head(5)
        .to_string(index=False)
    )

    print()
    print("✓ Validation passed")

    return True


def main():

    print("=" * 70)
    print("M1 PREPROCESSING VALIDATION")
    print("=" * 70)

    success = True

    for split, filename in FILES:

        result = validate_file(
            split,
            filename
        )

        if not result:
            success = False

    print()
    print("=" * 70)

    if success:
        print("ALL PREPROCESSING CHECKS PASSED")
    else:
        print("PREPROCESSING VALIDATION FAILED")

    print("=" * 70)

    return success


if __name__ == "__main__":
    raise SystemExit(
        0 if main() else 1
    )