import json
import logging
from pathlib import Path
from evaluation.datasets import DEEP_QA, MULTIHOP_QA, INTENT_QA, NEGATIVE_QA

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path("data/eval")


def write_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")
    log.info(f"Wrote {len(records):>3} records → {path}")


def main():
    write_jsonl(OUT_DIR / "deep_qa.jsonl",     DEEP_QA)
    write_jsonl(OUT_DIR / "multihop_qa.jsonl", MULTIHOP_QA)
    write_jsonl(OUT_DIR / "intent_qa.jsonl",   INTENT_QA)
    write_jsonl(OUT_DIR / "negative_qa.jsonl", NEGATIVE_QA)
    log.info("\nNext → run 03_generate_candidates.py (needs Qdrant running)")


if __name__ == "__main__":
    main()