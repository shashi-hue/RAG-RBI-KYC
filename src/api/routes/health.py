from fastapi import APIRouter
from pydantic import BaseModel
from qdrant_client import QdrantClient
from src.api.dependencies import get_cfg

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status:     str
    qdrant:     str
    collection: str
    vectors:    int


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready", response_model=HealthResponse)
def readiness():
    """Checks Qdrant is reachable and collection has vectors."""
    cfg    = get_cfg()
    qdrant = QdrantClient(url=cfg.qdrant.url,api_key=cfg.qdrant.api_key,timeout=30,)

    try:
        info   = qdrant.get_collection(cfg.embedding.collection_name)
        count  = info.points_count
        status = "ready" if count > 0 else "empty"
    except Exception as e:
        return HealthResponse(
            status="unavailable", qdrant=str(e),
            collection=cfg.embedding.collection_name, vectors=0,
        )

    return HealthResponse(
        status     = status,
        qdrant     = "connected",
        collection = cfg.embedding.collection_name,
        vectors    = count,
    )
