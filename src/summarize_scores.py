import csv
from pathlib import Path

path = Path("eval/manual_scores.csv")

fields = [
    "baseline_correctness",
    "rag_correctness",
    "rag_groundedness",
    "retrieval_relevance",
]

rows = []
with path.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

for field in fields:
    values = [float(r[field]) for r in rows if r[field] != ""]
    avg = sum(values) / len(values)
    print(f"{field}: {avg:.2f}")