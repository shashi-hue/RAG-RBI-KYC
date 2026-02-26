import json
import time
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams,
    PointStruct, SparseVector,
    HnswConfigDiff,
)

log = logging.getLogger(__name__)

CHUNKS_PATH   = "data/processed/chunks.jsonl"
ANNEX_IV_PATH = "data/processed/annex_iv.jsonl"
APPENDIX_PATH = "data/processed/repealed_circulars.json"
MANIFEST_PATH = "data/processed/embed_manifest.json"
DENSE_DIM     = 1024   # bge-large-en-v1.5


# Loaders


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Embedding helpers


def dense_embed(
    texts:       list[str],
    model:       SentenceTransformer,
    batch_size:  int,
) -> list[list[float]]:
    """Encode texts with bge-large. Prefix required by BGE for document encoding."""
    prefixed = [f"Represent this sentence: {t}" for t in texts]
    vecs = model.encode(
        prefixed,
        batch_size           = batch_size,
        show_progress_bar    = True,
        normalize_embeddings = True,   # unit vectors → cosine sim works correctly
    )
    return vecs.tolist()


def sparse_embed(
    texts: list[str],
    model: SparseTextEmbedding,
) -> list[SparseVector]:
    """BM25 sparse vectors via FastEmbed."""
    results = []
    for sv in model.embed(texts):
        results.append(
            SparseVector(
                indices = sv.indices.tolist(),
                values  = sv.values.tolist(),
            )
        )
    return results


# Qdrant collection


def ensure_collection(qdrant: QdrantClient, name: str):
    existing = {c.name for c in qdrant.get_collections().collections}
    if name in existing:
        log.info(f"Collection '{name}' already exists — upserting into it")
        return

    qdrant.create_collection(
        collection_name      = name,
        vectors_config       = {
            "dense": VectorParams(
                size        = DENSE_DIM,
                distance    = Distance.COSINE,
                hnsw_config = HnswConfigDiff(m=16, ef_construct=200),
            ),
        },
        sparse_vectors_config = {
            "sparse": SparseVectorParams(),   # BM25 index, no size needed
        },
    )
    log.info(f"Created hybrid collection '{name}'  (dense={DENSE_DIM}d + BM25 sparse)")


# Shared upsert helper


def build_and_upsert(
    qdrant:       QdrantClient,
    collection:   str,
    dense_model:  SentenceTransformer,
    sparse_model: SparseTextEmbedding,
    batch_size:   int,
    texts:        list[str],
    payloads:     list[dict],
    id_offset:    int,
    label:        str,
) -> int:
    assert len(texts) == len(payloads), "texts and payloads must be same length"

    log.info(f"[{label}] Dense encoding {len(texts)} texts...")
    d_vecs = dense_embed(texts, dense_model, batch_size)

    log.info(f"[{label}] Sparse (BM25) encoding {len(texts)} texts...")
    s_vecs = sparse_embed(texts, sparse_model)

    points = [
        PointStruct(
            id      = id_offset + i,
            vector  = {
                "dense":  d_vecs[i],
                "sparse": s_vecs[i],
            },
            payload = payloads[i],
        )
        for i in range(len(texts))
    ]

    qdrant.upsert(collection_name=collection, points=points, wait=True)
    log.info(f"[{label}] Done {len(points)} points upserted")
    return len(points)




def upsert_chunks(
    qdrant:       QdrantClient,
    collection:   str,
    dense_model:  SentenceTransformer,
    sparse_model: SparseTextEmbedding,
    batch_size:   int,
    id_offset:    int = 0,
) -> int:
    records  = load_jsonl(CHUNKS_PATH)
    texts    = [r["embed_text"] for r in records]
    payloads = [
        {
            "chunk_id":       r["chunk_id"],
            "source":         "chunks",
            "chapter":        r["chapter"],
            "chapter_title":  r.get("chapter_title", ""),
            "part":           r.get("part"),
            "paragraph":      r.get("paragraph"),
            "page":           r.get("page", 0),
            "status":         r["status"],
            "text":           r["text"],
            "citation":       r["citation"],
            "historical_text":r.get("historical_text"),
            "footnotes":      json.dumps(r.get("footnotes", [])),
        }
        for r in records
    ]
    return build_and_upsert(
        qdrant, collection, dense_model, sparse_model,
        batch_size, texts, payloads, id_offset, "chunks",
    )




def upsert_annex_iv(
    qdrant:       QdrantClient,
    collection:   str,
    dense_model:  SentenceTransformer,
    sparse_model: SparseTextEmbedding,
    batch_size:   int,
    id_offset:    int = 100_000,
) -> int:
    records  = load_jsonl(ANNEX_IV_PATH)
    texts    = [r["embed_text"] for r in records]
    payloads = [
        {
            "chunk_id":  r["chunk_id"],
            "source":    "annex_iv",
            "chapter":   "ANNEX_IV",
            "status":    "active",
            "row_label": r["row_label"],
            "row_data":  json.dumps(r["row_data"]),
            "citation":  r["citation"],
            "text":      r["row_label"],
        }
        for r in records
    ]
    return build_and_upsert(
        qdrant, collection, dense_model, sparse_model,
        batch_size, texts, payloads, id_offset, "annex_iv",
    )




def upsert_appendix(
    qdrant:       QdrantClient,
    collection:   str,
    dense_model:  SentenceTransformer,
    sparse_model: SparseTextEmbedding,
    batch_size:   int,
    id_offset:    int = 200_000,
) -> int:
    data    = load_json(APPENDIX_PATH)
    records = list(data.values())
    texts   = [
        f"Repealed circular {r['original']} dated {r['date']} — {r['repealed_by']}"
        for r in records
    ]
    payloads = [
        {
            "source":      "appendix",
            "chapter":     "APPENDIX",
            "status":      "repealed",
            "circular_no": r["original"],
            "date":        r["date"],
            "repealed_by": r["repealed_by"],
            "text":        texts[i],
            "citation":    f"{r['original']} dated {r['date']}",
        }
        for i, r in enumerate(records)
    ]
    return build_and_upsert(
        qdrant, collection, dense_model, sparse_model,
        batch_size, texts, payloads, id_offset, "appendix",
    )




@hydra.main(config_path="../../", config_name="params", version_base=None)
def main(cfg: DictConfig):
    t0 = time.time()

    log.info(f"Loading dense model:  {cfg.embedding.model}")
    dense_model  = SentenceTransformer(cfg.embedding.model)

    log.info("Loading sparse model: {Qdrant/bm25}")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    qdrant = QdrantClient(url=cfg.qdrant.url)
    collection = cfg.embedding.collection_name
    ensure_collection(qdrant, collection)

    bs = cfg.embedding.batch_size
    counts = {
        "chunks":   upsert_chunks(qdrant, collection, dense_model, sparse_model, bs),
        "annex_iv": upsert_annex_iv(qdrant, collection, dense_model, sparse_model, bs),
        "appendix": upsert_appendix(qdrant, collection, dense_model, sparse_model, bs),
    }
    counts["total"] = sum(counts.values())

    manifest = {
        "dense_model":  cfg.embedding.model,
        "sparse_model": "Qdrant/bm25",
        "collection":   collection,
        "vector_dim":   DENSE_DIM,
        "counts":       counts,
        "elapsed_sec":  round(time.time() - t0, 2),
        "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    Path(MANIFEST_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"Embedding complete — {counts['total']} vectors in {manifest['elapsed_sec']}s")
    log.info(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
