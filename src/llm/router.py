import re
import logging
from enum import Enum
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

log = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    FPI_DOCS    = "fpidocs"     # exhaustive FPI document listing
    CHAPTER     = "chapter"      # chapter-scoped query
    HISTORICAL  = "historical"   # audit / deleted provision queries
    GENERAL     = "general"      # everything else


# Rule-based signals 

_FPI_DOCS_RE = re.compile(
    r'\b(fpi|foreign\s+portfolio\s+invest\w*)\b.{0,60}'
    r'\b(document|kyc|submit|require|need|list|annex)\b',
    re.I
)

_HISTORICAL_RE = re.compile(
    r'\b(deleted?|old|previous|former|repealed?|earlier|was\s+it|used\s+to|history|audit|before)\b',
    re.I
)

_CHAPTER_MAP = {
    r'\b(customer\s+acceptance|acceptance\s+policy)\b':    "III",
    r'\b(risk\s+(management|categor|based))\b':            "IV",
    r'\b(customer\s+identification|CIP)\b':                "V",
    r'\b(CDD|customer\s+due\s+diligence|EDD|enhanced)\b':  "VI",
    r'\b(record\s*(management|keeping|maintenance))\b':    "VII",
    r'\b(FIU|financial\s+intelligence|STR|CTR|report)\b':  "VIII",
    r'\b(wire\s+transfer|correspondent\s+bank)\b':         "X",
    r'\b(preliminary|definition|means)\b':                  "I",
}


# LLM fallback classifier 

_ROUTER_SYSTEM = """\
You are a query classifier for an RBI KYC regulatory RAG system.
Classify the user query into EXACTLY one of these intents:

fpi_docs   — user wants a COMPLETE LIST of KYC documents required for FPI (Foreign Portfolio Investor)
historical — user asks about deleted, repealed, old, or former provisions
chapter    — query is clearly about one specific chapter/topic area
general    — anything else

Reply with ONLY the intent label. No explanation.\
"""

_ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _ROUTER_SYSTEM),
    ("human",  "{query}"),
])


class QueryRouter:

    def __init__(self, llm: ChatGroq):
        self._llm_chain = _ROUTER_PROMPT | llm | StrOutputParser()

    def classify(self, query: str) -> tuple[QueryIntent, Optional[str]]:
        """
        Returns (intent, chapter_hint).
        chapter_hint is set only when intent == CHAPTER.

        Strategy:
          1. Run cheap regex rules first (no API call)
          2. Fall back to LLM only if rules are ambiguous
        """
        # Rule 1: FPI document listing
        if _FPI_DOCS_RE.search(query):
            log.debug(f"Router -> FPI_DOCS (rule)")
            return QueryIntent.FPI_DOCS, None

        # Rule 2: Historical / deleted provision
        if _HISTORICAL_RE.search(query):
            log.debug(f"Router -> HISTORICAL (rule)")
            return QueryIntent.HISTORICAL, None

        # Rule 3: Chapter-specific (keyword → chapter mapping)
        for pattern, chapter in _CHAPTER_MAP.items():
            if re.search(pattern, query, re.I):
                log.debug(f"Router -> CHAPTER {chapter} (rule)")
                return QueryIntent.CHAPTER, chapter

        # Rule 4: Explicit chapter mention  e.g. "Chapter VI" or "chapter 6"
        m = re.search(r'\bchapter\s+(VI{0,3}|I{1,4}|IV|V|IX|X[I]{0,2}|\d{1,2})\b', query, re.I)
        if m:
            roman = m.group(1).upper()
            log.debug(f"Router -> CHAPTER {roman} (explicit mention)")
            return QueryIntent.CHAPTER, roman

        # Fallback: LLM classification (only for ambiguous queries)
        log.debug(f"Router -> LLM fallback for: '{query}'")
        try:
            raw    = self._llm_chain.invoke({"query": query}).strip().lower()
            intent = QueryIntent(raw) if raw in QueryIntent._value2member_map_ else QueryIntent.GENERAL
            log.debug(f"Router <- LLM said: '{raw}' -> {intent}")
            return intent, None
        except Exception as e:
            log.warning(f"Router LLM fallback failed: {e} — defaulting to GENERAL")
            return QueryIntent.GENERAL, None
