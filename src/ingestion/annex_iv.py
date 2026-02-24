import json
import hashlib
import logging
from pathlib import Path
import pdfplumber
from .models import TableChunk
import re

log = logging.getLogger(__name__)

GROUP_HEADERS = [
    "entity level",
    "senior management",
    "whole time director",
    "authorized signator",
    "authorised signator",
    "ubo",
    "ultimate beneficial",
    "partner",           
    "trustee",           
    "principal officer", 
    "individual",        
]


def is_group_header(text: str) -> bool:
    t = text.lower().strip()
    return any(t.startswith(g) for g in GROUP_HEADERS)


def clean_cell(val) -> str:
    if val is None:
        return ""
    text = " ".join(str(val).split())
    # Only merge a trailing consonant-only fragment (e.g. "Memorandu m" → "Memorandum")
    # Vowel-containing words like "for", "of", "is" are never touched
    text = re.sub(r'(\w{5,})\s([b-df-hj-np-tv-z]{1,2})\b', r'\1\2', text)
    return text




def extract_annex_iv(pdf_path: str, out_path: str) -> list[TableChunk]:
    chunks        = []
    current_group = "Entity Level"
    real_header   = None
    header_idx    = None
    all_data_rows = []
    in_annex_iv   = False

    with pdfplumber.open(pdf_path) as pdf:
        for pg_num, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").lower()

            if "annex iv" in text and pg_num > 88:      # real Annex IV is page 94, TOC is page 2
                in_annex_iv = True

            if in_annex_iv and "appendix" in text and pg_num > 95:   # real Appendix is page 97
                break

            if not in_annex_iv:
                continue

            # Skip FPI category description page (page 96)
            if "eligible foreign investors" in text:
                continue

            for table in page.extract_tables():
                if not table:
                    continue
                for row in table:
                    all_data_rows.append([clean_cell(c) for c in row])

    # Find real header: row containing "Category I" and "Category II"
    for i, row in enumerate(all_data_rows):
        joined = " ".join(row).lower()
        if "category i" in joined and "category ii" in joined:
            real_header = row
            header_idx  = i
            break

    if real_header is None or header_idx is None:
        log.warning("Annex IV: header row not found — writing empty file")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("")
        return chunks

    # Map category column names → column indices
    # Use startswith — no regex, no backslash issues
    cat_cols = {}
    for idx, col in enumerate(real_header):
        if col.lower().strip().startswith("category"):
            cat_cols[col.strip()] = idx

    if not cat_cols:
        log.warning(f"Annex IV: no category columns found. Header was: {real_header}")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("")
        return chunks

    # Process data rows after header
    for row in all_data_rows[header_idx + 1:]:
        if not any(row):
            continue

        entity_col = row[0] if len(row) > 0 else ""
        doc_type   = row[1] if len(row) > 1 else ""

        if entity_col and is_group_header(entity_col):
            current_group = entity_col
            if not doc_type:
                continue
        elif entity_col and not doc_type:
            doc_type = entity_col

        skip_labels = ("document type", "fpi type", "")
        if not doc_type or doc_type.lower().strip() in skip_labels:
            continue

        row_data = {}
        for cat_name, col_idx in cat_cols.items():
            if col_idx < len(row):
                row_data[cat_name] = row[col_idx] or "—"

        if not row_data:
            continue

        row_label  = f"{current_group} — {doc_type}"
        row_label = re.sub(r'\s*\d{2,3}\s*$', '', row_label).strip()  # strip trailing fn markers
        row_label = re.sub(r'\s*@@\s*$', '', row_label).strip()        # strip @@
        row_label = re.sub(r'(\w)\s(\w)', lambda m: m.group(0) if ' ' in m.group(0) else m.group(0), row_label) 
        embed_text = (
            f"[Annex IV FPI KYC] {row_label}: "
            + " | ".join(f"{k}: {v}" for k, v in row_data.items())
        )  # ← closing paren

        cid = hashlib.sha256(row_label.encode()).hexdigest()[:12]

        tc = TableChunk(
            chunk_id  = cid,
            source    = "Annex IV",
            row_label = row_label,
            row_data  = row_data,
            embed_text= embed_text,
            citation  = f"{row_label}, Annex IV, Master Direction KYC 2016",
            footnotes = [],
        )  # ← closing paren
        chunks.append(tc)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    log.info(f"Annex IV: wrote {len(chunks)} rows to {out_path}")
    return chunks
