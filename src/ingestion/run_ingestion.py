import json
import time
import logging
import yaml
import mlflow
import sys
from pathlib import Path

from .parser    import parse_document, save_chunks
from .annex_iv  import extract_annex_iv
from .appendix  import parse_appendix
from .definitions import split_definitions_chunk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("ingestion")

def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)["ingestion"]


def main():
    cfg = load_params()   # ← everything comes from params.yaml now

    PDF_PATH     = cfg["pdf_path"]
    CHUNKS_OUT   = cfg["chunks_out"]
    ANNEX_IV_OUT = cfg["annex_iv_out"]
    REPEALED_OUT = cfg["repealed_out"]
    METRICS_OUT  = cfg["metrics_out"]

    mlflow.set_experiment("kyc-rag")

    with mlflow.start_run(run_name="ingestion"):

        # log params to mlflow too
        mlflow.log_params(cfg)

        t0 = time.time()

        # Step 1 — Main document
        log.info("Step 1/3 — Parsing main document...")
        chunks = parse_document(PDF_PATH)
        # ── Explode Para 3 definitions into per-term chunks ──────────────────────
        expanded: list = []
        for c in chunks:
            if c.paragraph == "3":
                expanded.extend(split_definitions_chunk(c))
            else:
                expanded.append(c)
        chunks = expanded
        log.info(f"After definition split: {len(chunks)} total chunks")

        save_chunks(chunks, CHUNKS_OUT)

        deleted  = sum(1 for c in chunks if c.status == "deleted")
        amended  = sum(1 for c in chunks if c.status == "amended")
        inserted = sum(1 for c in chunks if c.status == "inserted")
        with_fns = sum(1 for c in chunks if c.footnotes)

        # Step 2 — Annex IV
        log.info("Step 2/3 — Extracting Annex IV FPI table...")
        try:
            table_chunks = extract_annex_iv(PDF_PATH, ANNEX_IV_OUT)
        except Exception as e:
            log.warning(f"Annex IV failed: {e}")
            table_chunks = []

        # Step 3 — Appendix
        log.info("Step 3/3 — Parsing Appendix...")
        try:
            repealed = parse_appendix(PDF_PATH, REPEALED_OUT)
        except Exception as e:
            log.warning(f"Appendix failed: {e}")
            repealed = {}

        elapsed = round(time.time() - t0, 2)

        metrics = {
            "total_chunks":          len(chunks),
            "deleted_chunks":        deleted,
            "amended_chunks":        amended,
            "inserted_chunks":       inserted,
            "chunks_with_footnotes": with_fns,
            "table_rows":            len(table_chunks),
            "repealed_circulars":    len(repealed),
            "elapsed_sec":           elapsed,
        }

        # MLflow: log metrics
        mlflow.log_metrics(metrics)

        # MLflow: log output files as artifacts
        mlflow.log_artifact(CHUNKS_OUT,   artifact_path="processed")
        mlflow.log_artifact(ANNEX_IV_OUT, artifact_path="processed")
        mlflow.log_artifact(REPEALED_OUT, artifact_path="processed")

        # DVC metrics file
        Path(METRICS_OUT).parent.mkdir(parents=True, exist_ok=True)
        with open(METRICS_OUT, "w") as f:
            json.dump(metrics, f, indent=2)

        log.info("=== Done ===")
        log.info(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()