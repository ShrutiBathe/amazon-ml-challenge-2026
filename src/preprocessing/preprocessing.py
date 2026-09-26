import re
import unicodedata
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = DATASET_DIR / "processed"

CHUNK_SIZE = 100_000


TRAIN_FILES = [
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv",
]

TEST_FILES = [
    "test_source1.tsv",
    "test_source2.tsv",
    "test_source3.tsv",
]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_unicode(text):
    """Unicode normalization using NFKC."""
    if not isinstance(text, str):
        return ""

    return unicodedata.normalize("NFKC", text)


def normalize_text(text):
    """
    General normalization.

    Keeps Unicode characters so multilingual names
    such as Hindi, Tamil, French, etc. are preserved.
    """
    if not isinstance(text, str):
        return ""

    text = normalize_unicode(text)

    text = text.lower()

    # Normalize ampersand
    text = text.replace("&", " and ")

    # Replace punctuation/symbols with spaces
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def compact_text(text):
    """Remove spaces from normalized text."""
    if not text:
        return ""

    return re.sub(r"[^\w]", "", text, flags=re.UNICODE)


# ============================================================
# LEGAL SUFFIX NORMALIZATION
# ============================================================

def remove_legal_suffixes(name):
    """
    Remove legal suffixes only when they occur at the END
    of the business name.

    Multiple suffixes can be removed:
        ABC Pvt Ltd
        ABC Private Limited
        ABC Inc LLC
    """

    if not name:
        return ""

    text = name.strip()

    suffix_pattern = re.compile(
        r"""
        (?:
            \bprivate\s+limited\b |
            \bpublic\s+limited\b |
            \bprivate\s+ltd\b |
            \bpublic\s+ltd\b |
            \bpvt\s+ltd\b |
            \bpvt\b |
            \bltd\b |
            \blimited\b |
            \bllp\b |
            \bllc\b |
            \bincorporated\b |
            \bcorporation\b |
            \bcorp\b |
            \binc\b |
            \bcompany\b |
            \bco\b
        )
        \s*$
        """,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    previous = None
    current = text

    while current != previous:
        previous = current
        current = suffix_pattern.sub("", current).strip()

    return current


# ============================================================
# DUPLICATE TOKEN NORMALIZATION
# ============================================================

def deduplicate_tokens(text):
    """
    Remove consecutive duplicate tokens.

    Example:
        "abc abc restaurant" -> "abc restaurant"

    Only consecutive duplicates are removed.
    """
    if not text:
        return ""

    tokens = text.split()

    result = []

    for token in tokens:
        if not result or token != result[-1]:
            result.append(token)

    return " ".join(result)


# ============================================================
# RECORD TRANSFORMATION
# ============================================================

def transform_chunk(df):
    """Create derived representations for a dataframe chunk."""

    # Ensure expected columns exist
    for column in [
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ]:
        if column not in df.columns:
            df[column] = ""

    # Convert missing values safely
    df["business_name"] = df["business_name"].fillna("").astype(str)
    df["business_address"] = df["business_address"].fillna("").astype(str)
    df["country"] = df["country"].fillna("").astype(str)

    # --------------------------------------------------------
    # BUSINESS NAME
    # --------------------------------------------------------

    df["name_normalized"] = df["business_name"].map(normalize_text)

    df["name_compact"] = df["name_normalized"].map(compact_text)

    df["name_no_suffix"] = df["name_normalized"].map(
        remove_legal_suffixes
    )

    df["name_no_suffix_compact"] = df["name_no_suffix"].map(
        compact_text
    )

    df["name_dedup"] = df["name_normalized"].map(
        deduplicate_tokens
    )

    # --------------------------------------------------------
    # ADDRESS
    # --------------------------------------------------------

    df["address_normalized"] = df["business_address"].map(
        normalize_text
    )

    df["address_compact"] = df["address_normalized"].map(
        compact_text
    )

    # --------------------------------------------------------
    # COUNTRY
    # --------------------------------------------------------

    df["country_normalized"] = (
        df["country"]
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------------
    # BLOCKING HELPERS
    # --------------------------------------------------------

    df["name_prefix"] = (
        df["name_no_suffix_compact"]
        .str[:4]
    )

    df["name_prefix_6"] = (
        df["name_no_suffix_compact"]
        .str[:6]
    )

    df["name_first_token"] = (
        df["name_no_suffix"]
        .str.split()
        .str[0]
        .fillna("")
    )

    df["address_first_token"] = (
        df["address_normalized"]
        .str.split()
        .str[0]
        .fillna("")
    )

    return df


# ============================================================
# FILE PROCESSING
# ============================================================

def process_file(input_path, output_path):
    """
    Process a TSV file in chunks.

    This avoids loading multi-million-row files
    completely into RAM.
    """

    print()
    print("=" * 70)
    print(f"PROCESSING: {input_path.name}")
    print("=" * 70)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    first_chunk = True

    total_rows = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            input_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):

        processed = transform_chunk(chunk)

        processed.to_csv(
            output_path,
            sep="\t",
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk,
        )

        first_chunk = False

        total_rows += len(processed)

        print(
            f"  Chunk {chunk_number:>4}: "
            f"{len(processed):,} rows | "
            f"Total: {total_rows:,}"
        )

    print()
    print(f"Completed: {input_path.name}")
    print(f"Rows: {total_rows:,}")
    print(f"Output: {output_path}")


# ============================================================
# DATASET PROCESSING
# ============================================================

def process_dataset():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # TRAIN
    # ========================================================

    for filename in TRAIN_FILES:

        input_path = DATASET_DIR / "train" / filename
        output_path = OUTPUT_DIR / "train" / filename

        if output_path.exists():
            print(
                f"SKIPPING — already processed: "
                f"{output_path}"
            )
            continue

        if input_path.exists():

            process_file(
                input_path,
                output_path
            )

        else:

            print(
                f"WARNING — input not found: "
                f"{input_path}"
            )

    # ========================================================
    # TEST
    # ========================================================

    for filename in TEST_FILES:

        input_path = DATASET_DIR / "test" / filename
        output_path = OUTPUT_DIR / "test" / filename

        if output_path.exists():
            print(
                f"SKIPPING — already processed: "
                f"{output_path}"
            )
            continue

        if input_path.exists():

            process_file(
                input_path,
                output_path
            )

        else:

            print(
                f"WARNING — input not found: "
                f"{input_path}"
            )

    print()
    print("=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"Processed data: {OUTPUT_DIR}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    process_dataset()