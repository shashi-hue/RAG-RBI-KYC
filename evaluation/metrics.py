import math
import re
from typing import Optional


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def hit_rate(expected: list[str], retrieved: list[str]) -> float:
    """Did ANY expected chunk appear in retrieved set?"""
    return 1.0 if any(e in retrieved for e in expected) else 0.0


def mrr(expected: list[str], retrieved: list[str]) -> float:
    """Mean Reciprocal Rank — rank position of the first correct hit."""
    for i, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in expected:
            return round(1.0 / i, 4)
    return 0.0


def recall_at_k(expected: list[str], retrieved: list[str], k: int = 5) -> float:
    """What fraction of expected chunks appear in the top-k retrieved?"""
    if not expected:
        return 0.0
    hits = sum(1 for e in expected if e in retrieved[:k])
    return round(hits / len(expected), 4)


def precision_at_k(expected: list[str], retrieved: list[str], k: int = 5) -> float:
    """What fraction of top-k retrieved chunks are actually relevant?"""
    if not retrieved:
        return 0.0
    top_k = retrieved[:k]
    hits  = sum(1 for r in top_k if r in expected)
    return round(hits / len(top_k), 4)


# ── Answer quality metrics ────────────────────────────────────────────────────

REFUSAL_PATTERNS = [
    r"outside (the )?scope",
    r"out of scope",
    r"i cannot answer",
    r"not covered (by|in)",
    r"does not cover",
    r"doesn'?t cover",
    r"cannot find",
    r"not found in",
    r"no (relevant |specific )?information",
    r"i (don'?t|do not) have",
    r"no relevant provisions",
]
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.I)


def is_refusal(answer: str) -> bool:
    """Return True if the answer text looks like a refusal / out-of-scope response."""
    return bool(_REFUSAL_RE.search(answer))


def answer_length_tokens(answer: str) -> int:
    """Rough token count (whitespace split)."""
    return len(answer.split())


# ── Aggregation helpers ───────────────────────────────────────────────────────

def safe_mean(values: list[float]) -> Optional[float]:
    """Mean of a list, returns None for empty lists."""
    return round(sum(values) / len(values), 4) if values else None