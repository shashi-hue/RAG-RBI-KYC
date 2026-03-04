# scripts/save_expected.py
import json
from pathlib import Path

TARGETS = [
    Path("data/eval/faq_raw_qa.jsonl"),
    Path("data/eval/deep_qa.jsonl"),
    Path("data/eval/multihop_qa.jsonl"),
]

backup = {}

for path in TARGETS:
    if not path.exists():
        continue

    records = [
        json.loads(l)
        for l in path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    for r in records:
        qid = r.get("id") or r.get("question_id")
        expected = r.get("expected_chunk_ids", [])

        if qid and expected:
            backup[qid] = expected

Path("data/eval/backup_expected.json").write_text(
    json.dumps(backup, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(f"Saved {len(backup)} expected_chunk_id entries to backup_expected.json")