import re
import time
import json
import logging
from typing import Iterator, Optional

import hydra
from omegaconf import DictConfig
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

from src.retrieval.retriever import KYCRetriever, RetrievedChunk
from src.llm.prompts import KYC_PROMPT
from src.llm.response import KYCResponse, CitationSource
from src.llm.router import QueryRouter, QueryIntent

log = logging.getLogger(__name__)


# Context formatter 

def format_context_numbered(chunks: list[RetrievedChunk]) -> str:
    """
    Converts retrieved chunks into a numbered reference list for the LLM prompt.
    Numbers correspond to [1], [2] citations in the answer.
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=1):

        # Header line
        header = f"[{i}] {chunk.citation}"
        if chunk.status == "deleted":
            header += "  ⚠️ DELETED PROVISION"
        elif chunk.status in ("amended", "inserted"):
            header += f"  [{chunk.status.upper()}]"

        # Body
        if chunk.status == "deleted" and chunk.historical_text:
            body = f"[Former text, no longer in force]: {chunk.historical_text}"
        elif chunk.source == "annex_iv" and chunk.row_data:
            rows = "\n".join(f"  • {cat}: {val}" for cat, val in chunk.row_data.items())
            body = f"{chunk.text}\n{rows}"
        else:
            body = chunk.text

        blocks.append(f"{header}\n{body}")

    return "\n\n---\n\n".join(blocks)


# Citation extractor

_REF_RE = re.compile(r'\[(\d+)\]')


def extract_cited_refs(answer: str) -> list[int]:
    """Pull all unique [N] reference numbers actually used in the answer."""
    return sorted(set(int(m) for m in _REF_RE.findall(answer)))


def build_citations(
    answer: str,
    chunks: list[RetrievedChunk],
) -> list[CitationSource]:
    """
    Map [N] numbers in the answer back to the corresponding RetrievedChunk.
    Only includes refs that actually appear in the answer text.
    """
    used_refs = extract_cited_refs(answer)
    citations = []
    for ref_num in used_refs:
        idx = ref_num - 1   # [1] -> index 0
        if 0 <= idx < len(chunks):
            chunk = chunks[idx]
            citations.append(CitationSource(
                ref_num      = ref_num,
                source       = chunk.source,
                chapter      = chunk.chapter,
                paragraph    = chunk.paragraph,
                citation     = chunk.citation,
                status       = chunk.status,
                score        = chunk.score,
                text_snippet = chunk.text[:200],
            ))
    return citations


# Main chain class

class KYCChain:

    def __init__(self, cfg: DictConfig):
        log.info(f"Initialising KYCChain  LLM={cfg.llm.model}  provider={cfg.llm.provider}")

        self.retriever = KYCRetriever(cfg)

        self.llm = ChatGroq(
            model       = cfg.llm.model,
            temperature = cfg.llm.temperature,
            max_tokens  = cfg.llm.max_tokens,
            api_key     = cfg.llm.api_key,
        )

        # LCEL chain: prompt → LLM → string output
        self._chain = KYC_PROMPT | self.llm | StrOutputParser()

        self.router = QueryRouter(self.llm)

        log.info("KYCChain ready.")


    # Internal: retrieve + build context

    def _get_chunks(
        self,
        query:          str,
        chapter:        Optional[str]       = None,
        sources:        Optional[list[str]] = None,
        include_deleted:bool                = False,
    ) -> list[RetrievedChunk]:
        if include_deleted:
            return self.retriever.retrieve(query, chapter=chapter, sources=sources)
        return self.retriever.retrieve_active(query, chapter=chapter, sources=sources)


    # Public: synchronous invoke 

    def invoke(
        self,
        query:           str,
        chapter:         Optional[str]       = None,
        sources:         Optional[list[str]] = None,
        include_deleted: bool                = False,
    ) -> KYCResponse:
        """
        Full RAG pipeline: retrieve → format → LLM → parse citations.
        Returns a structured KYCResponse.
        """
        t0     = time.time()
        chunks = self._get_chunks(query, chapter, sources, include_deleted)

        if not chunks:
            return KYCResponse(
                query       = query,
                answer      = "No relevant provisions found in the KYC Master Direction "
                              "for this query.",
                citations   = [],
                elapsed_sec = round(time.time() - t0, 2),
            )

        context = format_context_numbered(chunks)
        answer  = self._chain.invoke({"context": context, "query": query})

        citations = build_citations(answer, chunks)

        return KYCResponse(
            query                  = query,
            answer                 = answer,
            citations              = citations,
            has_deleted_provisions = any(c.status == "deleted"  for c in citations),
            has_amended_provisions = any(c.status in ("amended", "inserted") for c in citations),
            chunks_used            = len(chunks),
            elapsed_sec            = round(time.time() - t0, 2),
        )


    # Public: streaming invoke 

    def stream(
        self,
        query:           str,
        chapter:         Optional[str]       = None,
        sources:         Optional[list[str]] = None,
        include_deleted: bool                = False,
    ) -> Iterator[str]:
        """
        Streams answer tokens as they arrive from Groq.
        Use this for FastAPI SSE / WebSocket endpoints.

        Usage:
            for token in chain.stream(query):
                print(token, end="", flush=True)
        """
        chunks  = self._get_chunks(query, chapter, sources, include_deleted)
        if not chunks:
            yield "No relevant provisions found in the KYC Master Direction for this query."
            return

        context = format_context_numbered(chunks)
        for token in self._chain.stream({"context": context, "query": query}):
            yield token


    # Public: async invoke (for FastAPI)

    async def ainvoke(
        self,
        query:           str,
        chapter:         Optional[str]       = None,
        sources:         Optional[list[str]] = None,
        include_deleted: bool                = False,
    ) -> KYCResponse:
        """Async version of invoke() — use inside FastAPI route handlers."""
        t0     = time.time()
        chunks = self._get_chunks(query, chapter, sources, include_deleted)

        if not chunks:
            return KYCResponse(
                query       = query,
                answer      = "No relevant provisions found in the KYC Master Direction.",
                citations   = [],
                elapsed_sec = round(time.time() - t0, 2),
            )

        context = format_context_numbered(chunks)
        answer  = await self._chain.ainvoke({"context": context, "query": query})

        citations = build_citations(answer, chunks)

        return KYCResponse(
            query                  = query,
            answer                 = answer,
            citations              = citations,
            has_deleted_provisions = any(c.status == "deleted"  for c in citations),
            has_amended_provisions = any(c.status in ("amended", "inserted") for c in citations),
            chunks_used            = len(chunks),
            elapsed_sec            = round(time.time() - t0, 2),
        )


    # Convenience wrappers 

    def ask_fpi(self, query: str) -> KYCResponse:
        t0 = time.time()

        # All 18 Annex IV rows — instant scroll
        annex_chunks = self.retriever.fetch_all_by_source("annex_iv")

        # Para chunks — dense-only, NO reranker (they're just context)
        para_chunks  = self.retriever.retrieve(
            query,
            sources        = ["chunks"],
            exclude_status = ["deleted", "repealed"],
            skip_rerank    = True,   
        )[:3]

        # Annex IV rows first (direct answer), paragraphs after (context)
        merged = annex_chunks + para_chunks

        if not merged:
            return KYCResponse(
                query       = query,
                answer      = "No relevant provisions found.",
                citations   = [],
                elapsed_sec = round(time.time() - t0, 2),
            )

        context   = format_context_numbered(merged)
        answer    = self._chain.invoke({"context": context, "query": query})
        citations = build_citations(answer, merged)

        return KYCResponse(
            query                  = query,
            answer                 = answer,
            citations              = citations,
            has_deleted_provisions = any(c.status == "deleted"  for c in citations),
            has_amended_provisions = any(c.status in ("amended","inserted") for c in citations),
            chunks_used            = len(merged),
            elapsed_sec            = round(time.time() - t0, 2),
        )



    def ask_chapter(self, query: str, chapter: str) -> KYCResponse:
        """Restrict context to a specific chapter."""
        return self.invoke(query, chapter=chapter)

    def ask_with_history(self, query: str) -> KYCResponse:
        """Include deleted provisions — for compliance audit / historical queries."""
        return self.invoke(query, include_deleted=True)
    
    def query(self, user_input: str) -> KYCResponse:
        """
        Single unified entry point for all queries.
        Router classifies intent and dispatches to the right method.
        """
        intent, chapter_hint = self.router.classify(user_input)
        log.info(f"Router -> intent={intent.value}  chapter={chapter_hint}")

        if intent == QueryIntent.FPI_DOCS:
            return self.ask_fpi(user_input)

        elif intent == QueryIntent.HISTORICAL:
            return self.ask_with_history(user_input)

        elif intent == QueryIntent.CHAPTER and chapter_hint:
            return self.ask_chapter(user_input, chapter_hint)

        else:
            return self.invoke(user_input)


#  CLI entry point

@hydra.main(config_path="../../", config_name="params", version_base=None)
def main(cfg: DictConfig):
    chain = KYCChain(cfg)

    test_queries = [
        "What documents does a Category III FPI need to submit for entity level KYC?",
        "What are the conditions for small accounts under simplified KYC?",
        "What are the wire transfer reporting requirements for cross-border transactions?",
        "Is board resolution mandatory for Category I FPI?",
        "What was the old provision for customer identification before it was deleted?",
    ]

    for query in test_queries:
        print(f"\nQ: {query}")
        response = chain.query(query)   # ← always this, caller never decides
        print(response.to_terminal())


if __name__ == "__main__":
    main()
