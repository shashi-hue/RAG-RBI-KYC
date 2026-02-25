import logging
from functools import lru_cache
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from src.llm.chain import KYCChain

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_cfg() -> DictConfig:
    """Load Hydra config once — cached for app lifetime."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir="../../", version_base=None):
        return compose(config_name="config")


@lru_cache(maxsize=1)
def get_chain() -> KYCChain:
    """Singleton KYCChain — models load once at startup."""
    log.info("Initialising KYCChain singleton...")
    chain = KYCChain(get_cfg())
    log.info("KYCChain ready.")
    return chain
