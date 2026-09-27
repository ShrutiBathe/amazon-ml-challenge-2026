import pandas as pd

p = r"dataset/processed/m3_test/test_m3_predictions.tsv"

total = 0
high = 0
ents = set()
highents = set()

for c in pd.read_csv(
    p,
    sep="\t",
    usecols=["source1_id", "match_probability"],
    chunksize=500000,
):
    total += len(c)
    ents.update(c["source1_id"])
    h = c[c["match_probability"] >= 0.90]
    high += len(h)
    highents.update(h["source1_id"])

print("Total candidates:", total)
print("Candidates >=0.90:", high)
print("Entities with >=0.90:", len(highents))
print("All entities:", len(ents))
