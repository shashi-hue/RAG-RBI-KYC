import re
import hashlib
import json
from pathlib import Path
from typing import Optional
import fitz

from .models import Footnote, KYCChunk


# Regex patterns

# Inline superscript marker: digit(s) immediately before a capital letter/quote
MARKER_RE    = re.compile(r'(?<!\d)(\d{1,3})(?=[A-Z"\'(\u201c])')

CHAPTER_RE   = re.compile(r'^chapter\s+(I{1,4}|IV|V[I]{0,3}|IX|X[I]{0,2})\b', re.I)
PART_RE      = re.compile(r'^part\s+(I{1,4}|IV|V[I]{0,3})\b', re.I)
PARA_RE = re.compile(r'^(\d{1,3}[A-Z]?)\.\s+\S')
ANNEX_RE = re.compile(r'^annex\s+[IVX]+\s*(?:[-—–].*)?$', re.I)
APPENDIX_RE  = re.compile(r'^appendix\b', re.I)

# Footer: a line starting with a number followed by an amendment action word
FOOTER_RE    = re.compile(
    r'^\s*(\d{1,3})\s+(Inserted|Amended|Deleted|Substituted|Omitted)\b', re.I
)

# Individual footnote segment parser
FN_ENTRY_RE  = re.compile(
    r'(?m)^(\d{1,3})\s+(Inserted|Amended|Deleted|Substituted|Omitted)'
    r'(?:\s+vide)?\s+(.+?)(?=\n\d{1,3}\s+(?:Inserted|Amended|Deleted|Substituted|Omitted)|$)',
    re.DOTALL
)

CIRCULAR_RE  = re.compile(
    r'(?:circular\s+)?((DOR|DBR|DBOD)\.[A-Za-z]+\.[A-Za-z]+\.[^\s,;]+)',
    re.I
)
GAZETTE_RE   = re.compile(r'(G\.S\.R\.\s*\d+\([A-Z]\)[^\s,;]*)', re.I)
DATE_RE      = re.compile(
    r'dated\s+(\w+\s+\d{1,2},?\s*\d{4}|\d{1,2}\s+\w+,?\s*\d{4})',
    re.I
)
DELETED_TEXT_RE = re.compile(
    r'(?:read as|is as follows|portion read as)\s*[:\-]?\s*["\u201c]?(.+)',
    re.I | re.DOTALL
)
SHIFTED_RE = re.compile(r'shifted\s+to\s+paragraph\s+(\d+[A-Z]?)', re.I)

# TOC lines have 3+ consecutive dots — skip them everywhere
TOC_LINE_RE = re.compile(r'\.{3,}')


CHAPTER_TITLES = {
    "I":   "Preliminary",
    "II":  "General",
    "III": "Customer Acceptance Policy",
    "IV":  "Risk Management",
    "V":   "Customer Identification Procedure",
    "VI":  "Customer Due Diligence Procedure",
    "VII": "Record Management",
    "VIII":"Reporting Requirements to FIU-India",
    "IX":  "International Agreements and Communications",
    "X":   "Other Instructions",
    "XI":  "Repeal Provisions",
    "INTRO":    "Introduction",
    "ANNEX_I":  "Annex I — Digital KYC Process",
    "ANNEX_II": "Annex II — UAPA Section 51A Procedure",
    "ANNEX_III":"Annex III — Risk Indicators",
}


# Footnote parser 

def parse_footnote(fn_num: str, action: str, body: str) -> Footnote:
    """Parse one footnote segment into a Footnote object."""
    ref = None
    m = CIRCULAR_RE.search(body)
    if m:
        ref = m.group(1).strip()
    else:
        m = GAZETTE_RE.search(body)
        if m:
            ref = m.group(1).strip()

    date = None
    m = DATE_RE.search(body)
    if m:
        date = m.group(1).strip()

    deleted_text = None
    m = DELETED_TEXT_RE.search(body)
    if m:
        deleted_text = m.group(1).strip().rstrip('"\u201d').strip()[:500]

    shifted_to = None
    m = SHIFTED_RE.search(body)
    if m:
        shifted_to = m.group(1)

    return Footnote(
        fn_num=fn_num,
        action=action.capitalize(),
        ref=ref,
        date=date,
        deleted_text=deleted_text,
        shifted_to=shifted_to,
    )


def extract_all_footnotes(pdf_path: str) -> dict[str, Footnote]:
    """
    Pass 1: scan every page footer and collect all footnotes into one pool.
    Returns {fn_num_str: Footnote}.
    """
    doc = fitz.open(pdf_path)
    all_footer_text = ""

    for page in doc:
        lines = page.get_text("text").split("\n")
        # Find where footer starts (first line that looks like a footnote number)
        footer_start = len(lines)
        for i in range(len(lines) - 1, max(len(lines) - 50, 0), -1):
            if FOOTER_RE.match(lines[i]):
                footer_start = i
                # Keep scanning up — footer might start earlier
        all_footer_text += "\n" + "\n".join(lines[footer_start:])

    doc.close()

    pool: dict[str, Footnote] = {}
    for m in FN_ENTRY_RE.finditer(all_footer_text):
        fn_num = m.group(1)
        action = m.group(2)
        body   = m.group(3).strip()
        pool[fn_num] = parse_footnote(fn_num, action, body)

    return pool


# Chunk builder

def make_chunk_id(chapter: str, part: Optional[str], para: Optional[str], page: int) -> str:
    key = f"{chapter}|{part or ''}|{para or ''}|{page}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def build_chunk(
    buffer:    list[str],
    chapter:   str,
    part:      Optional[str],
    paragraph: Optional[str],
    page:      int,
    fn_pool:   dict[str, Footnote],
    ) -> Optional[KYCChunk]:

    if not buffer or not chapter:
        return None

    # Join lines and extract inline markers 
    raw = " ".join(l.strip() for l in buffer if l.strip())
    markers = MARKER_RE.findall(raw)
    clean = re.sub(r'\s{2,}', ' ', MARKER_RE.sub('', raw)).strip()

    # ── Check deleted BEFORE the length filter ────────────────────────────────
    _t = re.sub(r'^\d{1,3}[A-Z]?\.\s*', '', clean).strip()
    _t = re.sub(r'^\d+', '', _t).strip()
    _is_deleted = bool(re.match(r'^deleted\.?$', _t, re.I))

    # ── Length filter — but never discard a deleted paragraph ─────────────────
    if len(clean) < 15 and not _is_deleted:
        return None

    # Link markers to footnotes
    linked = [fn_pool[m] for m in markers if m in fn_pool]

    seen = set()
    linked = [fn for fn in linked if fn.fn_num not in seen and not seen.add(fn.fn_num)]

    # Determine status
    status = "active"
    historical = None
    shifted_to = None

    
    if _is_deleted:
        status = "deleted"
        for fn in linked:
            if fn.deleted_text and not historical:
                historical = fn.deleted_text
            if fn.shifted_to and not shifted_to:
                shifted_to = fn.shifted_to
    elif any(fn.action in ("Amended", "Substituted") for fn in linked):   # ← dedented 2 levels
        status = "amended"
    elif linked and all(fn.action == "Inserted" for fn in linked): 
        status = "inserted"



    # Citation 
    ch_title = CHAPTER_TITLES.get(chapter, chapter)
    # latest amendment date for citation suffix
    latest = None
    for fn in linked:
        if fn.date:
            latest = fn.date   # last one wins — good enough for the suffix

    citation_parts = []
    if paragraph:
        citation_parts.append(f"Para {paragraph}")
    if part:
        citation_parts.append(f"Chapter {chapter} {part}")
    else:
        citation_parts.append(f"Chapter {chapter}")
    citation_parts.append("Master Direction KYC 2016")
    if latest:
        citation_parts.append(f"Updated {latest}")
    citation = ", ".join(citation_parts)

    circular_refs = list({
        fn.ref for fn in linked
        if fn.ref and not fn.ref.startswith("G.S.R")  # exclude gazette
    })

    # Append to embed_text so BM25 can find them:
    body_for_embed = historical if status == "deleted" and historical else clean
    ref_suffix     = ("  [Amended via: " + ", ".join(circular_refs) + "]") if circular_refs else ""
    embed_text     = f"[{citation}] {body_for_embed}{ref_suffix}"



    return KYCChunk(
        chunk_id       = make_chunk_id(chapter, part, paragraph, page),
        chapter        = chapter,
        chapter_title  = ch_title,
        part           = part,
        paragraph      = paragraph,
        page           = page,
        text           = clean,
        embed_text     = embed_text,
        status         = status,
        historical_text= historical,
        footnotes      = [fn.to_dict() for fn in linked],
        citation       = citation,
    )


# Main parser 

def parse_document(pdf_path: str) -> list[KYCChunk]:
    """
    Two-pass parse:
      Pass 1 — collect all footnotes from all page footers
      Pass 2 — state machine over body lines, build chunks
    """
    # Pass 1
    fn_pool = extract_all_footnotes(pdf_path)

    # Pass 2
    doc     = fitz.open(pdf_path)
    chunks  = []

    chapter   = "INTRO"
    part      = None
    paragraph = None
    buffer    = []
    cur_page  = 1
    stop      = False   # True once we hit Annex IV or Appendix

    def flush():
        chunk = build_chunk(buffer, chapter, part, paragraph, cur_page, fn_pool)
        if chunk:
            chunks.append(chunk)
        buffer.clear()

    for pg_idx, page in enumerate(doc, start=1):
        cur_page = pg_idx
        if stop:
            break
        lines    = page.get_text("text").split("\n")

        # Find footer boundary
        footer_start = len(lines)
        for i in range(len(lines) - 1, max(len(lines) - 50, 0), -1):
            if FOOTER_RE.match(lines[i]):
                footer_start = i

        body = lines[:footer_start]

        for line in body:
            s = line.strip()
            if not s:
                continue

            if TOC_LINE_RE.search(s):
                continue

            # Only stop at Annex IV / Appendix when past actual content pages ─
            # TOC mentions these too — guard with page number
            if ANNEX_RE.match(s) and "iv" in s.lower() and cur_page > 70:
                flush()
                stop = True
            if APPENDIX_RE.match(s) and cur_page > 70:
                flush()
                stop = True
            if stop:
                continue


            # Detect Annex I/II/III — keep parsing but update chapter
            annex_m = ANNEX_RE.match(s)
            if annex_m and "iv" not in s.lower():
                flush()
                m = re.search(r'Annex\s+([IVX]+)', s, re.I)
                chapter = "ANNEX_" + m.group(1).upper() if m else "ANNEX_I"
                part      = None
                paragraph = None
                continue

            # Chapter
            ch_m = CHAPTER_RE.match(s)
            if ch_m:
                flush()
                chapter   = ch_m.group(1).upper()
                part      = None
                paragraph = None
                continue

            # Part
            pt_m = PART_RE.match(s)
            if pt_m:
                flush()
                part      = f"Part {pt_m.group(1).upper()}"
                paragraph = None
                continue

            # Paragraph
            para_m = PARA_RE.match(s)
            if para_m:
                new_para = para_m.group(1)

                # Guard: reject inline numbered sub-items (e.g. "1." "2." inside Para 3's
                # Beneficial Owner explanation). A real paragraph number never goes backward.
                try:
                    new_num = int(re.match(r'\d+', new_para).group())
                    cur_num = int(re.match(r'\d+', paragraph).group()) if (
                        paragraph and re.match(r'\d+', paragraph)
                    ) else 0
                    if new_num < cur_num:
                        buffer.append(s)   # treat as continuation, not a new paragraph
                        continue
                except (ValueError, AttributeError):
                    pass  # non-numeric para (e.g. "5A") — let it through normally

                flush()
                paragraph = new_para
                buffer.append(s)
                continue


            # Continuation of current paragraph
            buffer.append(s)

    flush()   # flush last buffer
    doc.close()
    return chunks


# Save

def save_chunks(chunks: list[KYCChunk], out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    return out_path