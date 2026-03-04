# scripts/restore_expected.py
import json
from pathlib import Path

TARGETS = [
    Path("data/eval/faq_raw_qa.jsonl"),
    Path("data/eval/deep_qa.jsonl"),
    Path("data/eval/multihop_qa.jsonl"),
]

backup = json.loads(
    Path("data/eval/backup_expected.json").read_text(encoding="utf-8")
)

restored = 0
not_found = []

for path in TARGETS:
    if not path.exists():
        continue

    records = [
        json.loads(l)
        for l in path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    updated = []

    for r in records:
        qid = r.get("id")

        if qid and qid in backup:
            r["expected_chunk_ids"] = backup[qid]
            restored += 1
        elif qid:
            not_found.append(qid)

        updated.append(r)

    path.write_text(
        "\n".join(json.dumps(rec, ensure_ascii=False) for rec in updated) + "\n",
        encoding="utf-8"
    )

    print(f"Updated {path.name}: {len(updated)} records")

print(f"\n✅ Restored expected_chunk_ids for {restored} questions")

if not_found:
    print(f"⚠️  {len(not_found)} question IDs not in backup: {not_found}")