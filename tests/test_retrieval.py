import hydra
from omegaconf import DictConfig
from src.retrieval.retriever import KYCRetriever

@hydra.main(config_path="../", config_name="params", version_base=None)
def main(cfg: DictConfig):
    retriever = KYCRetriever(cfg)

    test_queries = [
        ("natural language",  "what documents does a Category III FPI need to submit?"),
        ("exact term match",  "paragraph 23A small accounts"),
        ("circular ref",      "DOR.AML.REC.66/14.01.001/2023-24"),
        ("deleted provision", "wire transfer threshold"),
        ("annex iv specific", "Category II FPI board resolution requirement"),
    ]

    for label, query in test_queries:
        print(f"\n{'='*60}")
        print(f"[{label}]  {query}")
        print('='*60)
        results = retriever.retrieve_active(query)
        for r in results:
            print(f"  #{r.rank}  score={r.score:.4f}  [{r.source}] {r.citation}")
            print(f"       {r.text[:120]}...")

if __name__ == "__main__":
    main()
