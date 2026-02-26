import logging
from functools import lru_cache
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from src.llm.chain import KYCChain
from pathlib import Path

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_cfg() -> DictConfig:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    project_root = Path(__file__).resolve().parents[2]

    with initialize_config_dir(
        config_dir=str(project_root),
        version_base=None,
    ):
        return compose(config_name="params")


@lru_cache(maxsize=1)
def get_chain() -> KYCChain:
    """Singleton KYCChain — models load once at startup."""
    log.info("Initialising KYCChain singleton...")
    chain = KYCChain(get_cfg())
    log.info("KYCChain ready.")
    return chain
