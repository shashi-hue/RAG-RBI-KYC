import re
import hashlib
from typing import List
from .models import KYCChunk


# ── Actual section boundary patterns from the PDF ────────────────────────────
SECTION_A_RE = re.compile(
    r'\(a\)\s+Terms\s+bearing\s+meaning\s+assigned\s+in\s+terms',
    re.I
)
SECTION_B_RE = re.compile(
    r'\(b\)\s+Terms\s+bearing\s+meaning\s+assigned\s+in\s+this',
    re.I
)
SECTION_C_RE = re.compile(
    r'\(c\)\s+All\s+other\s+expressions',
    re.I
)

# Roman numeral sub-items — dot format, uppercase lookahead only (no re.I)
ROMAN_RE = re.compile(
    r'(?<!\w)'
    r'([ivxlIVXL]{1,6})\.'
    r'\s+'
    r'(?=[\d\u201c\u201e\u2018"A-Z])'
)

# Wire transfer sub-items inside 3(b)(xvii): "a. Term: ..."
WIRE_SUBITEM_RE = re.compile(
    r'\b([a-n])\.\s+([A-Z][A-Za-z\s\-]+?):'
)

# Anchored pattern to locate "xvii." inside section (b) text
WIRE_BLOCK_RE = re.compile(r'(?<!\w)xvii\.\s+')


def _make_def_chunk_id(section: str, roman: str) -> str:
    key = f"DEF|3|{section}|{roman}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _extract_term(text: str) -> str:
    """
    Extract the term name from a definition chunk.
    Handles quoted terms and unquoted terms like Beneficial Owner, Correspondent Banking.
    """
    # Quoted term — search first 300 chars (handles curly + straight quotes)
    m = re.search(
        r"""[""'\u201c\u2018\u201e]([^""'\u201d\u2019\u201c\u201e]{2,80})[""'\u201d\u2019']""",
        text[:300]
    )
    if m:
        return m.group(1).strip()

    # Unquoted term before "means", "refers", "is", "shall", or ":"
    m2 = re.match(
        r'^[ivxlIVXL]+\.\s+'
        r'(?:\d+[A-Za-z]?\s*)?'           # optional footnote marker e.g. "22A"
        r'([A-Z][A-Za-z\s\(\)\/\-]{1,60}?)'
        r'(?:\s+means\b|\s+refers\b|\s+is\b|\s+shall\b|[:\-–])',
        text, re.I
    )
    if m2:
        term = m2.group(1).strip()
        term = re.sub(r'\s+[a-z]\.$', '', term).strip()  # strip trailing " a." " b."
        return term

    # Last resort: first few words after roman numeral dot
    m3 = re.match(r'^[ivxlIVXL]+\.\s+(?:\d+[A-Za-z]?\s*)?((?:\S+\s+){1,3}\S+)', text)
    if m3:
        return m3.group(1).strip()[:60]

    return ""


def _split_wire_transfer(text: str, base_chunk: KYCChunk) -> List[KYCChunk]:
    """
    Split 3(b)(xvii) Wire transfer block by sub-items a–n.
    Each sub-item becomes its own chunk.
    """
    sub_matches = list(WIRE_SUBITEM_RE.finditer(text))
    if len(sub_matches) < 2:
        return []

    chunks = []
    for idx, match in enumerate(sub_matches):
        letter = match.group(1).lower()
        term_name = match.group(2).strip()
        start = match.start()
        end = sub_matches[idx + 1].start() if idx + 1 < len(sub_matches) else len(text)
        sub_text = text[start:end].strip()

        if len(sub_text) < 15:
            continue

        citation = (
            f"Para 3(b)(xvii)({letter}), Chapter {base_chunk.chapter}, "
            f"Master Direction KYC 2016"
        )
        embed_text = (
            f"[{citation}] "
            f"Definition of {term_name}. "
            f"{sub_text}"
        )
        chunks.append(KYCChunk(
            chunk_id        = _make_def_chunk_id("b_wire", letter),
            chapter         = base_chunk.chapter,
            chapter_title   = base_chunk.chapter_title,
            part            = base_chunk.part,
            paragraph       = f"3(b)(xvii)({letter})",
            page            = base_chunk.page,
            text            = sub_text,
            embed_text      = embed_text,
            status          = "active",
            historical_text = None,
            footnotes       = [],
            citation        = citation,
        ))
    return chunks


def _process_section(sec_letter: str, sec_text: str, chunk: KYCChunk,
                     result: List[KYCChunk]) -> None:
    """
    Process one section (a, b, or c) and append resulting chunks to result.
    For section 'b': wire transfer block (xvii) is extracted FIRST before
    ROMAN_RE runs, so its sub-items (i. j. k. l. ...) don't pollute 3(b) entries.
    """

    # ── 3(c) — single short paragraph ────────────────────────────────────────
    if sec_letter == 'c':
        citation = (
            f"Para 3(c), Chapter {chunk.chapter}, "
            f"Master Direction KYC 2016"
        )
        result.append(KYCChunk(
            chunk_id        = _make_def_chunk_id("c", "0"),
            chapter         = chunk.chapter,
            chapter_title   = chunk.chapter_title,
            part            = chunk.part,
            paragraph       = "3(c)",
            page            = chunk.page,
            text            = sec_text,
            embed_text      = f"[{citation}] {sec_text}",
            status          = "active",
            historical_text = None,
            footnotes       = [],
            citation        = citation,
        ))
        return

    # ── Section (b): carve out the xvii wire block BEFORE running ROMAN_RE ───
    wire_text = None
    scan_text = sec_text  # ROMAN_RE will only scan this (may be trimmed for section b)

    if sec_letter == 'b':
        wire_m = WIRE_BLOCK_RE.search(sec_text)
        if wire_m:
            scan_text = sec_text[:wire_m.start()]   # i–xvi only
            wire_text = sec_text[wire_m.start():]   # xvii block (a–n sub-items)

    # ── Find roman numeral boundaries in scan_text ────────────────────────────
    roman_matches = list(ROMAN_RE.finditer(scan_text))

    if len(roman_matches) < 2:
        # Fallback: keep whole section as one chunk
        citation = (
            f"Para 3({sec_letter}), Chapter {chunk.chapter}, "
            f"Master Direction KYC 2016"
        )
        result.append(KYCChunk(
            chunk_id        = _make_def_chunk_id(sec_letter, "0"),
            chapter         = chunk.chapter,
            chapter_title   = chunk.chapter_title,
            part            = chunk.part,
            paragraph       = f"3({sec_letter})",
            page            = chunk.page,
            text            = sec_text,
            embed_text      = f"[{citation}] {sec_text}",
            status          = "active",
            historical_text = None,
            footnotes       = [],
            citation        = citation,
        ))
        # Still process wire block if present
        if wire_text:
            _append_wire_chunks(wire_text, chunk, result)
        return

    # ── Split by each roman numeral entry ─────────────────────────────────────
    for r_idx, match in enumerate(roman_matches):
        roman = match.group(1).lower()
        r_start = match.start()
        r_end = (
            roman_matches[r_idx + 1].start()
            if r_idx + 1 < len(roman_matches)
            else len(scan_text)
        )
        def_text = scan_text[r_start:r_end].strip()

        if len(def_text) < 15:
            continue

        # Skip deleted entries e.g. "xvii. Deleted."
        if re.match(r'^[ivxlIVXL]+\.\s+Deleted\.?$', def_text, re.I):
            continue

        term = _extract_term(def_text)
        citation = (
            f"Para 3({sec_letter})({roman}), Chapter {chunk.chapter}, "
            f"Master Direction KYC 2016"
        )
        embed_text = (
            f"[{citation}] Definition of {term}. {def_text}"
            if term
            else f"[{citation}] {def_text}"
        )

        result.append(KYCChunk(
            chunk_id        = _make_def_chunk_id(sec_letter, roman),
            chapter         = chunk.chapter,
            chapter_title   = chunk.chapter_title,
            part            = chunk.part,
            paragraph       = f"3({sec_letter})({roman})",
            page            = chunk.page,
            text            = def_text,
            embed_text      = embed_text,
            status          = "active",
            historical_text = None,
            footnotes       = [],
            citation        = citation,
        ))

    # ── Append wire transfer chunks after i–xvi ───────────────────────────────
    if wire_text:
        _append_wire_chunks(wire_text, chunk, result)


def _append_wire_chunks(wire_text: str, chunk: KYCChunk,
                        result: List[KYCChunk]) -> None:
    """Split wire block into sub-chunks or keep as one fallback chunk."""
    wire_chunks = _split_wire_transfer(wire_text, chunk)
    if wire_chunks:
        result.extend(wire_chunks)
    else:
        citation = (
            f"Para 3(b)(xvii), Chapter {chunk.chapter}, "
            f"Master Direction KYC 2016"
        )
        result.append(KYCChunk(
            chunk_id        = _make_def_chunk_id("b", "xvii"),
            chapter         = chunk.chapter,
            chapter_title   = chunk.chapter_title,
            part            = chunk.part,
            paragraph       = "3(b)(xvii)",
            page            = chunk.page,
            text            = wire_text.strip(),
            embed_text      = f"[{citation}] {wire_text.strip()}",
            status          = "active",
            historical_text = None,
            footnotes       = [],
            citation        = citation,
        ))


def split_definitions_chunk(chunk: KYCChunk) -> List[KYCChunk]:
    """
    Splits Para 3 chunk into per-definition KYCChunks.
    3(a)(i)–(xxi): ~20 active definitions
    3(b)(i)–(xvi): 16 definitions + 3(b)(xvii)(a–n): wire transfer sub-items
    3(c): 1 catch-all chunk
    Returns [original chunk] if structure doesn't match expectations.
    """
    text = chunk.text

    ma = SECTION_A_RE.search(text)
    mb = SECTION_B_RE.search(text)
    mc = SECTION_C_RE.search(text)

    if not ma:
        return [chunk]  # unexpected structure — return unchanged

    sections = []
    if ma: sections.append((ma.start(), 'a'))
    if mb: sections.append((mb.start(), 'b'))
    if mc: sections.append((mc.start(), 'c'))
    sections.sort(key=lambda x: x[0])

    result: List[KYCChunk] = []

    for sec_idx, (sec_start, sec_letter) in enumerate(sections):
        sec_end = (
            sections[sec_idx + 1][0]
            if sec_idx + 1 < len(sections)
            else len(text)
        )
        sec_text = text[sec_start:sec_end].strip()
        _process_section(sec_letter, sec_text, chunk, result)

    return result if result else [chunk]


if __name__ == "__main__":
    test_lines = [
        'i. "Aadhaar number" shall have the meaning...',
        'iv. Beneficial Owner (BO) a. Where the customer...',
        'viii. 11"Digital KYC" means the capturing live photo...',
        'xvi. "Person" has the same meaning assigned in the Act...',
        'xvii. 20Deleted.',
        'xix. "Suspicious transaction" means a transaction...',
        "xx. 22A 'Small Account' means a savings account...",
    ]
    for line in test_lines:
        m = ROMAN_RE.match(line)
        term = _extract_term(line)
        print(f"ROMAN match: {bool(m):5}  term: '{term:40}'  line: {line[:50]}")
