import json

circular_refs = set()
with open("data/processed/chunks.jsonl", encoding="utf-8") as f:
    for line in f:
        chunk = json.loads(line)
        for fn in chunk.get("footnotes", []):
            if fn.get("ref"):
                circular_refs.add(fn["ref"])

for ref in sorted(circular_refs):
    print(ref)
