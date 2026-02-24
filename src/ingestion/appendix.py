import re
import json
from pathlib import Path
import pdfplumber
from .models import Footnote

REPEALED_BY = "Master Direction DBR.AML.BC.No.81/14.01.001/2015-16 dated February 25, 2016"


def _normalize_circular(raw: str) -> str:
    """
    Normalize circular number for consistent lookup.
    Removes internal spaces: 'DBOD. AML.BC. No.43/...' → 'DBOD.AML.BC.No.43/...'
    """
    # Remove spaces that appear around dots and slashes
    s = re.sub(r'\s*\.\s*', '.', raw.strip())
    s = re.sub(r'\s*/\s*', '/', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _is_circular_number(text: str) -> bool:
    """Must contain '/' and start with a known RBI dept prefix."""
    if "/" not in text:
        return False
    prefixes = (
        "DBOD", "DBR", "DOR", "UBD", "RPCD",
        "DNBS", "IDMD", "DBS", "DCBR", "DGBA"
    )
    return any(text.upper().startswith(p) for p in prefixes)


def parse_appendix(pdf_path: str, out_path: str) -> dict:
    """
    Extract all repealed circular numbers from the Appendix (pages 97–107).
    Returns {normalized_circular_no: {original, date, repealed_by}}.

    Key insight from debug: pdfplumber treats the first row of each page's
    table as the "header". Since every page continues the same table,
    the first row IS data (not a header) for pages 98+.
    Solution: process ALL rows of every table, including table[0].
    """
    lookup      = {}
    in_appendix = False

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text  = page.extract_text() or ""
            lower = text.lower()

            # Detect appendix start
            if not in_appendix:
                if "appendix" in lower or ("circulars" in lower and "repealed" in lower):
                    in_appendix = True

            if not in_appendix:
                continue

            for table in page.extract_tables():
                if not table:
                    continue

                # Process EVERY row (table[0] is also data on continuation pages)
                for row in table:
                    if not row or len(row) < 2:
                        continue

                    cells = [re.sub(r'\s+', ' ', str(c or "")).strip() for c in row]

                    # Find the circular number cell — it's whichever cell has "/"
                    # and looks like a real RBI circular
                    circular_raw = None
                    date_raw     = ""

                    for i, cell in enumerate(cells):
                        if _is_circular_number(cell):
                            circular_raw = cell
                            # Date is usually the next non-empty cell
                            for j in range(i + 1, len(cells)):
                                if cells[j] and not _is_circular_number(cells[j]):
                                    date_raw = cells[j]
                                    break
                            break

                    if not circular_raw:
                        continue

                    # For partially repealed circulars, the cell may contain
                    # "DBOD.BP.BC.57/21.01.001/95 – Paragraph 2(b)"
                    # Keep original for display but normalize key
                    circular_key = _normalize_circular(
                        circular_raw.split("–")[0].split("-Para")[0].strip()
                    )

                    if circular_key not in lookup:
                        lookup[circular_key] = {
                            "original":    circular_raw,
                            "date":        date_raw[:40],
                            "repealed_by": REPEALED_BY,
                        }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(lookup, f, indent=2)

    return lookup


def is_repealed(circular_no: str, lookup: dict) -> str | None:
    """
    Pre-retrieval guard. Normalizes the query circular number before lookup.
    Returns a repeal notice string or None.
    """
    key = _normalize_circular(circular_no)
    if key in lookup:
        rec = lookup[key]
        return (
            f"⚠ Note: Circular {rec['original']} (dated {rec['date']}) "
            f"has been repealed by the {rec['repealed_by']}. "
            f"Please refer to the current Master Direction instead."
        )
    return None
