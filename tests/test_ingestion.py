import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.ingestion.parser import (
    parse_footnote, build_chunk, make_chunk_id,
    extract_all_footnotes, parse_document,
    MARKER_RE,
)
from src.ingestion.annex_iv import clean_cell, is_group_header, extract_annex_iv
from src.ingestion.models import Footnote, KYCChunk

PDF_PATH = "data/raw/KYC_Directions_2025.pdf"
CHUNKS_PATH = "data/processed/chunks.jsonl"
ANNEX_IV_PATH = "data/processed/annex_iv.jsonl"
APPENDIX_PATH = "data/processed/appendix.json"


# ─────────────────────────────────────────────
# Unit tests — parse_footnote
# ─────────────────────────────────────────────

class TestParseFootnote:
    def test_inserted_with_circular_and_date(self):
        body = "vide DOR.AML.BC.No.27/14.01.001/2019-20 dated January 9, 2020"
        fn = parse_footnote("11", "Inserted", body)
        assert fn.action == "Inserted"
        assert fn.ref == "DOR.AML.BC.No.27/14.01.001/2019-20"
        assert fn.date == "January 9, 2020"
        assert fn.deleted_text is None

    def test_amended_with_date(self):
        body = "vide DOR.AML.REC.44/14.01.001/2023-24 dated October 17, 2023"
        fn = parse_footnote("35", "Amended", body)
        assert fn.action == "Amended"
        assert fn.date == "October 17, 2023"

    def test_deleted_with_text(self):
        body = 'portion read as "The customer shall submit OVD within 30 days"'
        fn = parse_footnote("99", "Deleted", body)
        assert fn.deleted_text is not None
        assert "OVD" in fn.deleted_text

    def test_shifted_to(self):
        body = "shifted to paragraph 23A vide DOR.AML.BC.No.27/14.01.001/2019-20"
        fn = parse_footnote("42", "Amended", body)
        assert fn.shifted_to == "23A"

    def test_no_circular_no_date(self):
        fn = parse_footnote("1", "Inserted", "Some generic text with no reference")
        assert fn.ref is None
        assert fn.date is None


# ─────────────────────────────────────────────
# Unit tests — MARKER_RE
# ─────────────────────────────────────────────

class TestMarkerRE:
    def test_finds_superscript_before_capital(self):
        markers = MARKER_RE.findall("This is15Amended by circular")
        assert "15" in markers

    def test_ignores_standalone_numbers(self):
        markers = MARKER_RE.findall("paragraph 23 of Chapter VI")
        assert markers == []

    def test_multiple_markers(self):
        markers = MARKER_RE.findall("The12Amended provision13Inserted here")
        assert "12" in markers
        assert "13" in markers


# ─────────────────────────────────────────────
# Unit tests — build_chunk status logic
# ─────────────────────────────────────────────

def make_fn(action: str, num: str = "1") -> Footnote:
    return Footnote(fn_num=num, action=action, ref=None, date="January 1, 2020",
                    deleted_text=None, shifted_to=None)

class TestBuildChunkStatus:
    def _chunk(self, text, fns):
        pool = {fn.fn_num: fn for fn in fns}
        # inject markers manually by making text contain marker pattern
        return build_chunk([text], "I", None, "5", 10, pool)

    def test_active_no_footnotes(self):
        c = build_chunk(["5. This is a normal active paragraph."], "I", None, "5", 10, {})
        assert c is not None
        assert c.status == "active"

    def test_deleted_short_text_not_filtered(self):
        c = build_chunk(["71. Deleted"], "VI", None, "71", 55, {})
        assert c is not None
        assert c.status == "deleted"

    def test_amended_beats_inserted(self):
        """If ANY fn is Amended, status must be amended — not inserted."""
        pool = {
            "10": make_fn("Amended", "10"),
            "11": make_fn("Inserted", "11"),
        }
        # Build raw text that triggers both markers
        c = build_chunk(
            ["5. This10Amended paragraph was also11Inserted with new text added here."],
            "II", None, "5", 10, pool
        )
        assert c is not None
        assert c.status == "amended"

    def test_all_inserted_gives_inserted(self):
        pool = {"11": make_fn("Inserted", "11")}
        c = build_chunk(
            ["6. This11Inserted is a wholly new paragraph added to the directions."],
            "II", None, "6", 11, pool
        )
        assert c is not None
        assert c.status == "inserted"

    def test_length_filter_kills_noise(self):
        c = build_chunk(["Page 3"], "I", None, None, 3, {})
        assert c is None

    def test_deleted_bypasses_length_filter(self):
        c = build_chunk(["22. Deleted"], "VI", None, "22", 40, {})
        assert c is not None
        assert c.status == "deleted"


# ─────────────────────────────────────────────
# Unit tests — clean_cell / is_group_header
# ─────────────────────────────────────────────

class TestCleanCell:
    def test_none_returns_empty(self):
        assert clean_cell(None) == ""

    def test_newlines_collapsed(self):
        assert clean_cell("Mandatory\nfor all") == "Mandatory for all"

    def test_multiple_spaces(self):
        assert clean_cell("Proof   of   Identity") == "Proof of Identity"

    def test_broken_word(self):
        # "Memorandu m" should become "Memorandum"
        result = clean_cell("Memorandu m and Articles")
        assert "Memorandu m" not in result

class TestIsGroupHeader:
    def test_entity_level(self):
        assert is_group_header("Entity Level") is True

    def test_ubo(self):
        assert is_group_header("Ultimate Beneficial Owner (UBO)") is True

    def test_authorized_signatories(self):
        assert is_group_header("Authorised Signatories") is True

    def test_non_header(self):
        assert is_group_header("Proof of Identity") is False

    def test_case_insensitive(self):
        assert is_group_header("ENTITY LEVEL") is True


# ─────────────────────────────────────────────
# Integration tests — require actual PDF
# ─────────────────────────────────────────────

@pytest.mark.skipif(not Path(PDF_PATH).exists(), reason="PDF not available")
class TestIntegration:

    def test_footnote_pool_size(self):
        """Should extract at least 150 footnotes from the PDF."""
        pool = extract_all_footnotes(PDF_PATH)
        assert len(pool) >= 150, f"Only {len(pool)} footnotes found"

    def test_parse_document_total_chunks(self):
        chunks = parse_document(PDF_PATH)
        assert 100 <= len(chunks) <= 130, f"Unexpected chunk count: {len(chunks)}"

    def test_no_duplicate_chunk_ids(self):
        chunks = parse_document(PDF_PATH)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk_ids found"

    def test_deleted_chunks_exist(self):
        chunks = parse_document(PDF_PATH)
        deleted = [c for c in chunks if c.status == "deleted"]
        assert len(deleted) >= 5, f"Expected ≥5 deleted chunks, got {len(deleted)}"

    def test_amended_beats_inserted_in_output(self):
        """No chunk should be 'inserted' if it also has an Amended footnote."""
        chunks = parse_document(PDF_PATH)
        for c in chunks:
            actions = {fn["action"] for fn in c.footnotes}
            if "Amended" in actions or "Substituted" in actions:
                assert c.status != "inserted", (
                    f"Chunk {c.chunk_id} (para {c.paragraph}) has Amended fn "
                    f"but status=inserted"
                )

    def test_chapter_labels_valid(self):
        valid = {"INTRO","I","II","III","IV","V","VI","VII","VIII","IX","X","XI",
                 "ANNEX_I","ANNEX_II","ANNEX_III"}
        chunks = parse_document(PDF_PATH)
        bad = {c.chapter for c in chunks if c.chapter not in valid}
        assert not bad, f"Invalid chapter labels: {bad}"


# ─────────────────────────────────────────────
# Integration tests — processed output files
# ─────────────────────────────────────────────

@pytest.mark.skipif(not Path(CHUNKS_PATH).exists(), reason="chunks.jsonl not built")
class TestProcessedChunks:

    def _load(self):
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    def test_no_duplicate_chunk_ids(self):
        data = self._load()
        ids = [d["chunk_id"] for d in data]
        dupes = [i for i in set(ids) if ids.count(i) > 1]
        assert not dupes, f"Duplicate chunk_ids: {dupes}"

    def test_all_required_fields_present(self):
        required = {"chunk_id","chapter","text","embed_text","status","citation","footnotes"}
        for row in self._load():
            missing = required - row.keys()
            assert not missing, f"chunk {row['chunk_id']} missing: {missing}"

    def test_status_values_valid(self):
        valid = {"active", "amended", "inserted", "deleted"}
        for row in self._load():
            assert row["status"] in valid, f"Bad status '{row['status']}' in {row['chunk_id']}"

    def test_deleted_chunks_have_text(self):
        for row in self._load():
            if row["status"] == "deleted":
                assert row["text"], f"Deleted chunk {row['chunk_id']} has no text"

    def test_embed_text_not_empty(self):
        for row in self._load():
            assert len(row["embed_text"]) > 20, f"Short embed_text in {row['chunk_id']}"


@pytest.mark.skipif(not Path(ANNEX_IV_PATH).exists(), reason="annex_iv.jsonl not built")
class TestProcessedAnnexIV:

    def _load(self):
        with open(ANNEX_IV_PATH, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    def test_row_count(self):
        rows = self._load()
        assert 15 <= len(rows) <= 25, f"Unexpected Annex IV rows: {len(rows)}"

    def test_no_duplicate_chunk_ids(self):
        rows = self._load()
        ids = [r["chunk_id"] for r in rows]
        dupes = [i for i in set(ids) if ids.count(i) > 1]
        assert not dupes, f"Duplicate Annex IV chunk_ids: {dupes}"

    def test_all_three_categories_present(self):
        for row in self._load():
            cats = set(row["row_data"].keys())
            assert "Category I" in cats
            assert "Category II" in cats
            assert "Category III" in cats

    def test_no_footnote_markers_in_labels(self):
        """Row labels must not end with raw footnote numbers like 'PAN 165'."""
        import re
        for row in self._load():
            label = row["row_label"]
            assert not re.search(r'\s\d{2,3}$', label), \
                f"Footnote marker in label: '{label}'"

    def test_no_broken_words_in_labels(self):
        for row in self._load():
            label = row["row_label"]
            # A broken word has a space inside what should be one word
            assert "du m" not in label, f"Broken word in label: '{label}'"


@pytest.mark.skipif(not Path(APPENDIX_PATH).exists(), reason="appendix.json not built")
class TestProcessedAppendix:

    def _load(self):
        with open(APPENDIX_PATH, encoding="utf-8") as f:
            return json.load(f)

    def test_circular_count(self):
        data = self._load()
        assert len(data) == 255, f"Expected 255 repealed circulars, got {len(data)}"

    def test_all_have_date_and_repealed_by(self):
        for key, val in self._load().items():
            assert val.get("date"), f"Missing date for {key}"
            assert val.get("repealed_by"), f"Missing repealed_by for {key}"
