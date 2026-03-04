import re
import hashlib
from typing import List
from .models import KYCChunk

# ── Actual section boundary patterns from the PDF ────────────────────────────
# 3(a): "Terms bearing meaning assigned in terms of Prevention of Money-Laundering..."
SECTION_A_RE = re.compile(
    r'\(a\)\s+Terms\s+bearing\s+meaning\s+assigned\s+in\s+terms',
    re.I
)
# 3(b): "Terms bearing meaning assigned in this Directions..."
SECTION_B_RE = re.compile(
    r'\(b\)\s+Terms\s+bearing\s+meaning\s+assigned\s+in\s+this',
    re.I
)
# 3(c): "All other expressions unless defined herein..."
SECTION_C_RE = re.compile(
    r'\(c\)\s+All\s+other\s+expressions',
    re.I
)

# Handles both "quoted terms" and Beneficial Owner (BO), Correspondent Banking, etc.
ROMAN_RE = re.compile(
    r'(?<!\()'                         
    r'\b([ivxl]{1,6})\.'              
    r'\s+'                             # space(s)
    r'(?=["\u201c\u2018\u201eA-Z0-9])' # lookahead: quote or uppercase/digit next
)

# Wire transfer sub-items inside 3(b)(xvii): "a. Term: ..."
WIRE_SUBITEM_RE = re.compile(
    r'\b([a-n])\.\s+([A-Z][A-Za-z\s\-]+?):',
)


def _make_def_chunk_id(section: str, roman: str) -> str:
    key = f"DEF|3|{section}|{roman}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _extract_term(text: str) -> str:
    """
    Extract the term name from a definition chunk.
    Handles both quoted terms: "Aadhaar number" means...
    And unquoted terms: Beneficial Owner (BO) / Correspondent Banking:
    """
    # Quoted term — search first 300 chars
    m = re.search(
        r'["\u201c\u2018\u201e]([^"\u201d\u2019\u201c\u201e]{2,80})["\u201d\u2019]',
        text[:300]
    )
    if m:
        return m.group(1).strip()

    # Unquoted term before "means", "refers", "is", or ":"
    m2 = re.match(
        r'^\([ivxlIVXL]+\)\s+([A-Z][A-Za-z\s\(\)\/\-]{1,60}?)'
        r'(?:\s+means\b|\s+refers\b|\s+is\b|\s+shall\b|[:\-–])',
        text, re.I
    )
    if m2:
        return m2.group(1).strip()

    # Last resort: first 4 words after roman numeral
    m3 = re.match(r'^\([ivxlIVXL]+\)\s+((?:\S+\s+){1,3}\S+)', text)
    if m3:
        return m3.group(1).strip()[:60]

    return ""


def _split_wire_transfer(text: str, base_chunk: KYCChunk) -> List[KYCChunk]:
    """
    Further split 3(b)(xvii) Wire transfer definitions by sub-items a–n.
    Each sub-item (a. Batch transfer, b. Beneficiary, ...) becomes its own chunk.
    """
    sub_matches = list(WIRE_SUBITEM_RE.finditer(text))
    if len(sub_matches) < 2:
        return []  # Can't split, caller keeps whole chunk

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
            chunk_id        = _make_def_chunk_id(f"b_wire", letter),
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


def split_definitions_chunk(chunk: KYCChunk) -> List[KYCChunk]:
    """
    Splits Para 3 chunk into per-definition KYCChunks.
    3(a)(i)–(xxi): 21 definitions (xvii is deleted — skipped)
    3(b)(i)–(xvii): 17 definitions (xvii wire transfer further split a–n)
    3(c): 1 chunk (short catch-all paragraph)
    Returns [original chunk] if structure doesn't match expectations.
    """
    text = chunk.text

    # ── Locate section boundaries ─────────────────────────────────────────────
    ma = SECTION_A_RE.search(text)
    mb = SECTION_B_RE.search(text)
    mc = SECTION_C_RE.search(text)

    if not ma:
        # Para 3 text doesn't have expected structure — return original unchanged
        return [chunk]

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

        # ── 3(c) — single short paragraph ────────────────────────────────────
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
            continue

        # ── 3(a) and 3(b): split by roman numeral sub-items ──────────────────
        roman_matches = list(ROMAN_RE.finditer(sec_text))

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
            continue

        for r_idx, match in enumerate(roman_matches):
            roman = match.group(1).lower()
            r_start = match.start()
            r_end = (
                roman_matches[r_idx + 1].start()
                if r_idx + 1 < len(roman_matches)
                else len(sec_text)
            )
            def_text = sec_text[r_start:r_end].strip()

            if len(def_text) < 15:
                continue

            # Skip deleted entries like "(xvii) Deleted"
            if re.match(r'^[ivxl]+\.\s+Deleted\.?$', def_text, re.I):
                continue

            # ── Special case: 3(b)(xvii) Wire transfer — split further ────────
            if sec_letter == 'b' and roman == 'xvii':
                wire_chunks = _split_wire_transfer(def_text, chunk)
                if wire_chunks:
                    result.extend(wire_chunks)
                    continue
                # If split failed, fall through to keep whole (xvii) as one chunk

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

    return result if result else [chunk]
