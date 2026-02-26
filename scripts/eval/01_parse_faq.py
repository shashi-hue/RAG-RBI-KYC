"""
evaluation/01_parse_faq.py
Parses FAQ-s_KYC_Directions_2025.pdf → data/eval/faq_raw_qa.jsonl (37 records).
"""

import re
import json
import logging
from pathlib import Path
import pdfplumber

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

FAQ_PDF  = Path("data/raw/FAQ-s_KYC_Directions_2025.pdf")
OUT_PATH = Path("data/eval/faq_raw_qa.jsonl")

# Q 1. / Q1. / Q. 1 — all variants in this PDF
Q_RE = re.compile(r'^Q\.?\s*(\d+)\.\s*(.*)', re.I)
A_RE = re.compile(r'^Ans\.?\s*(.*)', re.I)


def parse_faq(pdf_path: Path) -> list[dict]:
    pairs, q_id, q_text, a_lines = [], None, None, []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for line in (page.extract_text() or "").splitlines():
                line = line.strip()
                if not line:
                    continue

                q_m = Q_RE.match(line)
                a_m = A_RE.match(line)

                if q_m:
                    if q_text and a_lines:
                        pairs.append({"faq_num": int(q_id),
                                      "question": q_text.strip(),
                                      "reference_answer": " ".join(a_lines).strip()})
                    q_id, q_text, a_lines = q_m.group(1), q_m.group(2), []

                elif a_m and q_text is not None:
                    a_lines = [a_m.group(1)]

                elif q_text is not None and not a_lines:
                    q_text += " " + line          # question continuation

                elif q_text is not None and a_lines:
                    a_lines.append(line)           # answer continuation

    if q_text and a_lines:
        pairs.append({"faq_num": int(q_id),
                      "question": q_text.strip(),
                      "reference_answer": " ".join(a_lines).strip()})
    return pairs


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pairs = parse_faq(FAQ_PDF)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for p in pairs:
            record = {
                "id":               f"faq_{p['faq_num']:03d}",
                "type":             "faq",
                "question":         p["question"],
                "reference_answer": p["reference_answer"],
                "expected_chunk_ids": [],   # ← fill after 03_generate_candidates.py
            }
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    log.info(f"Parsed {len(pairs)} FAQ pairs → {OUT_PATH}")
    log.info("Next → run 02_write_datasets.py then 03_generate_candidates.py")


if __name__ == "__main__":
    main()