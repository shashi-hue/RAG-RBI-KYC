import hydra
from omegaconf import DictConfig
from src.llm.chain import KYCChain

@hydra.main(config_path="../", config_name="params", version_base=None)
def main(cfg: DictConfig):
    chain = KYCChain(cfg)

    test_queries = [
        "What documents does a Category III FPI need to submit for entity level KYC?",
        # "What are the conditions for small accounts under simplified KYC?",
        # "What are the wire transfer reporting requirements for cross-border transactions?",
        # "Is board resolution mandatory for Category I FPI?",
        # "What was the old provision for customer identification before it was deleted?",
    ]

    for query in test_queries:
        print(f"\nQ: {query}")
        response = chain.query(query)   # ← always this, caller never decides
        print(response.to_terminal())


if __name__ == "__main__":
    main()