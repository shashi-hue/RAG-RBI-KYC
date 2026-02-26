import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

SOURCES = [
    Path("data/eval/faq_raw_qa.jsonl"),
    Path("data/eval/deep_qa.jsonl"),
    Path("data/eval/multihop_qa.jsonl"),
    Path("data/eval/intent_qa.jsonl"),
    Path("data/eval/negative_qa.jsonl"),
]
OUT_PATH = Path("data/eval/eval_dataset.jsonl")
RETRIEVAL_TYPES = {"faq", "deep", "multihop"}
STRIP_KEYS      = {"_candidates", "hint", "requires_chunks", "hint_chapter"}


def main():
    all_records, counts = [], {}

    for src in SOURCES:
        if not src.exists():
            log.warning(f"  SKIP (not found): {src}")
            continue

        with open(src, encoding="utf-8") as f:
            records = [json.loads(l) for l in f if l.strip()]

        kept = 0
        for r in records:
            t = r.get("type", "unknown")
            if t in RETRIEVAL_TYPES and not r.get("expected_chunk_ids"):
                continue     # skip unlabelled
            for k in STRIP_KEYS:
                r.pop(k, None)
            all_records.append(r)
            kept += 1
            counts[t] = counts.get(t, 0) + 1

        log.info(f"  {src.name}: kept {kept}/{len(records)}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in all_records:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")

    log.info(f"\neval_dataset.jsonl → {len(all_records)} total records")
    for t, n in sorted(counts.items()):
        log.info(f"  {t:<12} {n}")
    log.info("\nNext → run 05_run_eval.py")


if __name__ == "__main__":
    main()