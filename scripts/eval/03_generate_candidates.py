import json
import logging
from pathlib import Path
from src.api.dependencies import get_cfg
from src.retrieval.retriever import KYCRetriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

TOP_K   = 5
TARGETS = [
    Path("data/eval/faq_raw_qa.jsonl"),
    Path("data/eval/deep_qa.jsonl"),
    Path("data/eval/multihop_qa.jsonl"),
]


def main():
    cfg       = get_cfg()
    retriever = KYCRetriever(cfg)

    for path in TARGETS:
        if not path.exists():
            log.warning(f"SKIP (not found): {path}")
            continue

        with open(path, encoding="utf-8") as f:
            records = [json.loads(l) for l in f if l.strip()]

        for rec in records:
            chunks = retriever.retrieve_active(rec["question"])[:TOP_K]
            rec["_candidates"] = [
                {
                    "rank":      c.rank,
                    "chunk_id":  c.payload.get("chunk_id", ""),
                    "citation":  c.citation,
                    "chapter":   c.payload.get("chapter", ""),
                    "paragraph": c.payload.get("paragraph", ""),
                    "status":    c.status,
                    "score":     c.score,
                    "snippet":   c.text[:160],
                }
                for c in chunks
            ]
            log.info(f"[{rec['id']}] top candidate: {chunks[0].citation if chunks else 'none'}")

        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                json.dump(r, f, ensure_ascii=False)
                f.write("\n")

        log.info(f"Updated {len(records)} records → {path}")

    log.info("\n MANUAL STEP: fill expected_chunk_ids from _candidates in each .jsonl")
    log.info("Then run 04_merge_dataset.py")


if __name__ == "__main__":
    main()