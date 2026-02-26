import json
import math
import logging
from dataclasses import dataclass, field
from typing import Optional
import torch

from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, MatchAny,
    Prefetch, FusionQuery, Fusion,
    SparseVector,
)

log = logging.getLogger(__name__)


# Return type


@dataclass
class RetrievedChunk:
    rank:      int
    score:     float
    source:    str          # "chunks" | "annex_iv" | "appendix"
    chapter:   str
    status:    str
    text:      str
    citation:  str
    payload:   dict = field(repr=False)

    # Convenience accessors so callers don't dig into payload
    @property
    def paragraph(self) -> Optional[str]:
        return self.payload.get("paragraph")

    @property
    def part(self) -> Optional[str]:
        return self.payload.get("part")

    @property
    def historical_text(self) -> Optional[str]:
        return self.payload.get("historical_text")

    @property
    def footnotes(self) -> list[dict]:
        raw = self.payload.get("footnotes", "[]")
        return json.loads(raw) if isinstance(raw, str) else raw

    @property
    def row_data(self) -> Optional[dict]:
        raw = self.payload.get("row_data")
        if raw is None:
            return None
        return json.loads(raw) if isinstance(raw, str) else raw

    def to_context_block(self) -> str:
        """
        Formats chunk as a single string for LLM context injection.
        Includes citation, status warning for deleted chunks,
        and row_data table for Annex IV rows.
        """
        lines = [f"[{self.citation}]"]

        if self.status == "deleted":
            lines.append("⚠️  NOTE: This provision has been DELETED.")
            if self.historical_text:
                lines.append(f"Former text: {self.historical_text}")

        elif self.status in ("amended", "inserted"):
            lines.append(f"(Status: {self.status.upper()})")

        if self.source == "annex_iv" and self.row_data:
            lines.append(self.text)
            for cat, val in self.row_data.items():
                lines.append(f"  {cat}: {val}")
        else:
            lines.append(self.text)

        return "\n".join(lines)

# Retriever


class KYCRetriever:

    def __init__(self, cfg):
        log.info(f"Loading dense model:   {cfg.embedding.model}")
        self.dense_model  = SentenceTransformer(cfg.embedding.model)

        log.info("Loading sparse model:  Qdrant/bm25")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

        log.info(f"Loading reranker:      {cfg.reranker.model}")
        self.reranker = CrossEncoder(
            cfg.reranker.model,
            device="cuda" if torch.cuda.is_available() else "cpu",
            max_length=512,   # ← truncate to speed up CPU inference
        )

        self.qdrant         = QdrantClient(url=cfg.qdrant.url)
        self.collection     = cfg.embedding.collection_name
        self.top_k_retrieve = cfg.reranker.top_k_retrieve   # candidates before rerank
        self.top_k_return   = cfg.reranker.top_k_return     # final results after rerank

        log.info("KYCRetriever ready.")


    # Private: encode query

    def _dense_query(self, query: str) -> list[float]:
        """BGE requires the same prefix at query time too."""
        vec = self.dense_model.encode(
            [f"Represent this sentence: {query}"],
            normalize_embeddings=True,
        )
        return vec[0].tolist()

    def _sparse_query(self, query: str) -> SparseVector:
        sv = next(self.sparse_model.embed([query]))
        return SparseVector(
            indices=sv.indices.tolist(),
            values=sv.values.tolist(),
        )


    # Private: build Qdrant filter 
    @staticmethod
    def _build_filter(
        status_filter: Optional[str],
        chapter:       Optional[str],
        sources:       Optional[list[str]],
        exclude_status:Optional[list[str]],
    ) -> Optional[Filter]:
        must     = []
        must_not = []

        if status_filter:
            must.append(
                FieldCondition(key="status", match=MatchValue(value=status_filter))
            )
        if chapter:
            must.append(
                FieldCondition(key="chapter", match=MatchValue(value=chapter))
            )
        if sources:
            must.append(
                FieldCondition(key="source", match=MatchAny(any=sources))
            )
        if exclude_status:
            for s in exclude_status:
                must_not.append(
                    FieldCondition(key="status", match=MatchValue(value=s))
                )

        if not must and not must_not:
            return None
        return Filter(must=must or None, must_not=must_not or None)


    # Private: hybrid search

    def _hybrid_search(
        self,
        query:         str,
        qdrant_filter: Optional[Filter],
    ) -> list:
        """
        Hybrid search using qdrant-client ≥1.9 API.
        Named vectors are selected via `using=` parameter, not NamedVector wrapper.
        """
        dense_vec  = self._dense_query(query)
        sparse_vec = self._sparse_query(query)

        results = self.qdrant.query_points(
            collection_name = self.collection,
            prefetch        = [
                # Branch 1: dense cosine similarity
                Prefetch(
                    query  = dense_vec,        # plain list[float]
                    using  = "dense",          # ← which named vector to search
                    limit  = self.top_k_retrieve,
                    filter = qdrant_filter,
                ),
                # Branch 2: sparse BM25
                Prefetch(
                    query  = sparse_vec,       # SparseVector(indices, values)
                    using  = "sparse",         # ← which named vector to search
                    limit  = self.top_k_retrieve,
                    filter = qdrant_filter,
                ),
            ],
            query        = FusionQuery(fusion=Fusion.RRF),
            limit        = self.top_k_retrieve,
            with_payload = True,
        ).points

        return results



    # Private: rerank

    def _rerank(
        self,
        query:   str,
        results: list,
    ) -> list[tuple[float, object]]:
        """
        Cross-encoder reranking.
        Returns list of (score, qdrant_point) sorted descending.
        """
        if not results:
            return []

        texts  = [r.payload.get("text", "") for r in results]
        pairs  = [[query, t] for t in texts]
        scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(scores.tolist(), results),
            key     = lambda x: x[0],
            reverse = True,
        )
        return ranked[:self.top_k_return]

    def _sigmoid(self, x: float) -> float:
        return round(1 / (1 + math.exp(-x * 0.1)), 4) 
    # Public: main retrieve 


    def fetch_all_by_source(self, source: str) -> list[RetrievedChunk]:
        """
        Scroll ALL points for a given source — bypasses vector search
        and reranker entirely. Safe only for small bounded sets (annex_iv = 18 rows).
        """
        points, _ = self.qdrant.scroll(
            collection_name = self.collection,
            scroll_filter   = Filter(must=[
                FieldCondition(key="source", match=MatchValue(value=source))
            ]),
            limit        = 500,       # well above any source size
            with_payload = True,
            with_vectors = False,     # no vectors needed — not doing similarity
        )

        return [
            RetrievedChunk(
                rank     = i + 1,
                score    = 1.0,       # all rows equally fetched, no ranking
                source   = p.payload.get("source", ""),
                chapter  = p.payload.get("chapter", ""),
                status   = p.payload.get("status", ""),
                text     = p.payload.get("text", ""),
                citation = p.payload.get("citation", ""),
                payload  = p.payload,
            )
            for i, p in enumerate(points)
        ]


    def retrieve(
        self,
        query:          str,
        chapter:        Optional[str]       = None,
        status_filter:  Optional[str]       = None,
        exclude_status: Optional[list[str]] = None,
        sources:        Optional[list[str]] = None,
        skip_rerank: bool = False,
    ) -> list[RetrievedChunk]:
        """
        Full hybrid retrieval pipeline:
          dense + sparse → RRF fusion → cross-encoder rerank

        Args:
            query:          Natural language or keyword query
            chapter:        Restrict to a chapter  e.g. "VI", "ANNEX_IV"
            status_filter:  Exact status match      e.g. "active"
            exclude_status: Statuses to exclude     e.g. ["deleted", "repealed"]
            sources:        Source filter            e.g. ["chunks", "annex_iv"]

        Returns:
            List of RetrievedChunk, ranked by reranker score.
        """
        qdrant_filter = self._build_filter(
            status_filter, chapter, sources, exclude_status
        )

        # Step 1: hybrid vector search
        raw_results = self._hybrid_search(query, qdrant_filter)
        if not raw_results:
            log.warning(f"No results found for query: '{query}'")
            return []
        if skip_rerank:
            # Return top_k_return results in RRF order directly
            return [
                RetrievedChunk(
                    rank     = i + 1,
                    score    = 1.0,
                    source   = r.payload.get("source", ""),
                    chapter  = r.payload.get("chapter", ""),
                    status   = r.payload.get("status", ""),
                    text     = r.payload.get("text", ""),
                    citation = r.payload.get("citation", ""),
                    payload  = r.payload,
                )
                for i, r in enumerate(raw_results[:self.top_k_return])
            ]

        # Step 2: cross-encoder reranking
        ranked = self._rerank(query, raw_results)

        # Step 3: build return objects
        chunks = []
        for rank, (score, point) in enumerate(ranked, start=1):
            p = point.payload
            chunks.append(RetrievedChunk(
                rank     = rank,
                score    = self._sigmoid(float(score)),
                source   = p.get("source", ""),
                chapter  = p.get("chapter", ""),
                status   = p.get("status", ""),
                text     = p.get("text", ""),
                citation = p.get("citation", ""),
                payload  = p,
            ))

        return chunks


    # Public: convenience wrappers 

    def retrieve_active(self, query: str, **kwargs) -> list[RetrievedChunk]:
        """Retrieve only active + amended + inserted — skip deleted/repealed."""
        return self.retrieve(
            query,
            exclude_status=["deleted", "repealed"],
            **kwargs,
        )

    def retrieve_fpi_kyc(self, query: str) -> list[RetrievedChunk]:
        """Restrict to Annex IV FPI KYC table rows only."""
        return self.retrieve(
            query,
            sources=["annex_iv"],
        )

    def retrieve_chapter(self, query: str, chapter: str) -> list[RetrievedChunk]:
        """Restrict to a specific chapter. Excludes deleted."""
        return self.retrieve(
            query,
            chapter=chapter,
            exclude_status=["deleted", "repealed"],
        )

    def retrieve_with_deleted(self, query: str) -> list[RetrievedChunk]:
        """Include deleted paragraphs — for historical/audit queries."""
        return self.retrieve(query)   # no filter at all


    # Public: format context for LLM

    @staticmethod
    def format_context(chunks: list[RetrievedChunk]) -> str:
        """
        Formats retrieved chunks into a single context string
        ready to inject into a prompt.
        """
        blocks = []
        for chunk in chunks:
            blocks.append(f"--- Result {chunk.rank} (score: {chunk.score}) ---")
            blocks.append(chunk.to_context_block())
        return "\n\n".join(blocks)
