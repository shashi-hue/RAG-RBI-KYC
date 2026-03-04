import json
with open("data/processed/chunks.jsonl",encoding="utf-8") as f:
    chunks = [json.loads(l) for l in f if l.strip()]

special_paras = {"a","b","c","d","xv","xvi"}

para3 = [
    c for c in chunks
    if (
        str(c.get("paragraph","")).startswith("3(") or
        (c.get("chapter") == "I" and c.get("paragraph") in special_paras)
    )
]

for c in para3:
    print(f"para={c.get('paragraph',''):10}  id={c.get('chunk_id','')}  {c.get('text','')[:60]}")