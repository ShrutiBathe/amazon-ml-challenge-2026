from pathlib import Path
from collections import defaultdict
import pandas as pd

BASE = Path(__file__).resolve().parents[1]

S1 = BASE / "dataset" / "processed" / "test" / "test_source1.tsv"
S3 = BASE / "dataset" / "processed" / "test" / "test_source2.tsv"
OUT = BASE / "dataset" / "processed" / "m3_candidates" / "test_source2_exact_name.tsv"

CHUNK = 100_000
MAX_BLOCK = 300

print("=" * 70)
print("M3 TEST SOURCE2 EXACT-NAME CANDIDATE GENERATION")
print("=" * 70)

print("\nLoading Source1...")
s1 = pd.read_csv(
    S1,
    sep="\t",
    dtype=str,
    usecols=["entity_id", "name_compact"],
)

s1["name_compact"] = (
    s1["name_compact"]
    .fillna("")
    .astype(str)
    .str.strip()
)

s1 = s1[s1["name_compact"] != ""]

print(f"Source1 usable rows: {len(s1):,}")

# Build Source1 index: name_compact -> Source1 IDs
s1_index = defaultdict(list)

for entity_id, name in zip(
    s1["entity_id"],
    s1["name_compact"],
):
    s1_index[name].append(entity_id)

print(f"Unique Source1 names: {len(s1_index):,}")

if OUT.exists():
    OUT.unlink()

first_write = True
total = 0
usable_s3 = 0
valid_names = set()

print("\nScanning Source2...")

# First pass: count Source2 name frequencies.
from collections import Counter

counts = Counter()

for chunk in pd.read_csv(
    S3,
    sep="\t",
    dtype=str,
    usecols=["entity_id", "name_compact"],
    chunksize=CHUNK,
):
    names = (
        chunk["name_compact"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    names = names[names != ""]
    counts.update(names.tolist())

    usable_s3 += len(names)

    print(
        f"\r  scanned Source2 rows: {usable_s3:,}",
        end="",
        flush=True,
    )

valid_names = {
    name
    for name, count in counts.items()
    if count <= MAX_BLOCK
}

print()
print(f"Source2 usable name rows: {usable_s3:,}")
print(f"Valid name blocks <= {MAX_BLOCK}: {len(valid_names):,}")

# Second pass: generate candidates.
print("\nGenerating candidates...")

for chunk in pd.read_csv(
    S3,
    sep="\t",
    dtype=str,
    usecols=["entity_id", "name_compact"],
    chunksize=CHUNK,
):
    chunk["name_compact"] = (
        chunk["name_compact"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    chunk = chunk[
        chunk["name_compact"].isin(valid_names)
    ]

    if chunk.empty:
        continue

    records = []

    for candidate_id, name in zip(
        chunk["entity_id"],
        chunk["name_compact"],
    ):
        for source1_id in s1_index.get(name, []):
            records.append(
                (
                    source1_id,
                    candidate_id,
                    "source2",
                    "exact_name",
                )
            )

    if not records:
        continue

    result = pd.DataFrame(
        records,
        columns=[
            "source1_id",
            "candidate_id",
            "candidate_source",
            "blocking_strategy",
        ],
    )

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
        f"\r  generated candidates: {total:,}",
        end="",
        flush=True,
    )

print()
print("\n" + "=" * 70)
print(f"TOTAL SOURCE2 CANDIDATES: {total:,}")
print(f"OUTPUT: {OUT}")
print("=" * 70)

