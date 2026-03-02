import json
import time
import logging
from pathlib import Path
from collections import defaultdict

import mlflow

from src.api.dependencies import get_cfg, get_chain
from src.retrieval.retriever import KYCRetriever
from evaluation.metrics import (
    hit_rate, mrr, recall_at_k, precision_at_k,
    is_refusal, answer_length_tokens, safe_mean,
)
from evaluation.judge import score_answer   # single call -> faithfulness + answer_relevance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

EVAL_PATH    = Path("data/eval/eval_dataset.jsonl")
RESULTS_PATH = Path("data/eval/eval_results.jsonl")
METRICS_PATH = Path("data/eval/eval_metrics.json")


def main():
    if not EVAL_PATH.exists():
        raise FileNotFoundError(f"{EVAL_PATH} not found — run scripts/01->04 first.")

    cfg       = get_cfg()
    chain     = get_chain()        # KYCChain singleton — models load once
    retriever = KYCRetriever(cfg)  # raw retriever for chunk-level metrics

    with open(EVAL_PATH, encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    log.info(f"Loaded {len(dataset)} eval records from {EVAL_PATH}")

    results = []

    # Aggregators
    ret_hr, ret_mrr, ret_r5, ret_p5 = [], [], [], []
    ret_faith, ret_rel = [], []    # faithfulness + answer_relevance (both from one LLM call)
    intent_results = defaultdict(lambda: {"correct": 0, "total": 0})
    neg_refused, neg_total = 0, 0

    mlflow.set_experiment("kyc-rag-eval")

    with mlflow.start_run(run_name=f"eval_{int(time.time())}"):

        for item in dataset:
            q   = item["question"]
            typ = item.get("type", "unknown")
            t0  = time.time()

            # RETRIEVAL TYPES: faq / deep / multihop
            if typ in {"faq", "deep", "multihop"}:
                expected = item.get("expected_chunk_ids", [])

                # Raw retrieval for chunk-level metrics
                chunks        = retriever.retrieve_active(q)
                retrieved_ids = [c.payload.get("chunk_id", "") for c in chunks]

                # Full pipeline for answer-level metrics
                response = chain.query(q)

                # Retrieval metrics
                hr_ = hit_rate(expected, retrieved_ids)
                mrr_ = mrr(expected, retrieved_ids)
                r5_  = recall_at_k(expected, retrieved_ids, k=5)
                p5_  = precision_at_k(expected, retrieved_ids, k=5)

                # ONE LLM call -> both faithfulness and answer_relevance
                scores = score_answer(
                    question  = q,
                    reference = item.get("reference_answer", ""),
                    generated = response.answer,
                    llm       = chain.llm,
                )
                faith_ = scores["faithfulness"]
                rel_   = scores["answer_relevance"]

                # Accumulate
                ret_hr.append(hr_)
                ret_mrr.append(mrr_)
                ret_r5.append(r5_)
                ret_p5.append(p5_)
                if faith_ >= 0:
                    ret_faith.append(faith_)
                if rel_ >= 0:
                    ret_rel.append(rel_)

                result = {
                    "id":   item["id"],
                    "type": typ,
                    "question": q,
                    # retrieval
                    "expected_chunk_ids":  expected,
                    "retrieved_chunk_ids": retrieved_ids,
                    "hit_rate":            hr_,
                    "mrr":                 mrr_,
                    "recall_at_5":         r5_,
                    "precision_at_5":      p5_,
                    # answer quality (from single LLM call)
                    "faithfulness":        faith_,
                    "answer_relevance":    rel_,
                    # response
                    "answer":              response.answer,
                    "reference_answer":    item.get("reference_answer", ""),
                    "chunks_used":         response.chunks_used,
                    "has_deleted":         response.has_deleted_provisions,
                    "has_amended":         response.has_amended_provisions,
                    "answer_tokens":       answer_length_tokens(response.answer),
                    "elapsed_sec":         round(time.time() - t0, 3),
                }

                hr_icon = "+" if hr_ else "-"
                log.info(
                    f"[{item['id']:>10}] [{hr_icon}] "
                    f"HR={hr_:.0f} MRR={mrr_:.2f} R@5={r5_:.2f} P@5={p5_:.2f} "
                    f"Faith={faith_:.2f} Rel={rel_:.2f} | {q[:45]}..."
                )

            # INTENT
            elif typ == "intent":
                predicted_intent, chapter_hint = chain.router.classify(q)
                expected_intent = item.get("expected_intent", "")
                correct         = (predicted_intent.value == expected_intent)

                intent_results[expected_intent]["total"]   += 1
                intent_results[expected_intent]["correct"] += int(correct)

                result = {
                    "id":               item["id"],
                    "type":             typ,
                    "question":         q,
                    "expected_intent":  expected_intent,
                    "predicted_intent": predicted_intent.value,
                    "chapter_hint":     chapter_hint,
                    "correct":          correct,
                    "elapsed_sec":      round(time.time() - t0, 3),
                }

                icon = "+" if correct else "-"
                log.info(
                    f"[{item['id']:>10}] Intent [{icon}]  "
                    f"expected={expected_intent:<12} predicted={predicted_intent.value}"
                )

            # NEGATIVE
            elif typ == "negative":
                response = chain.query(q)
                refused  = is_refusal(response.answer)

                neg_refused += int(refused)
                neg_total   += 1

                result = {
                    "id":          item["id"],
                    "type":        typ,
                    "question":    q,
                    "refused":     refused,
                    "answer":      response.answer,
                    "elapsed_sec": round(time.time() - t0, 3),
                }

                icon = "REFUSED [+]" if refused else "ANSWERED [-]"
                log.info(f"[{item['id']:>10}] Negative: {icon}")

            else:
                log.warning(f"Unknown type for {item['id']} — skipped")
                continue

            results.append(result)

        # Build summary
        summary = {}

        if ret_hr:
            summary.update({
                "retrieval/n":                len(ret_hr),
                "retrieval/hit_rate":         safe_mean(ret_hr),
                "retrieval/mrr":              safe_mean(ret_mrr),
                "retrieval/recall_at_5":      safe_mean(ret_r5),
                "retrieval/precision_at_5":   safe_mean(ret_p5),
                "retrieval/faithfulness":     safe_mean(ret_faith) if ret_faith else -1,
                "retrieval/answer_relevance": safe_mean(ret_rel)   if ret_rel   else -1,
            })

        if intent_results:
            all_correct = sum(v["correct"] for v in intent_results.values())
            all_total   = sum(v["total"]   for v in intent_results.values())
            summary["intent/n"]                = all_total
            summary["intent/accuracy_overall"] = round(all_correct / all_total, 4) if all_total else 0
            for intent_name, counts in intent_results.items():
                summary[f"intent/accuracy_{intent_name}"] = (
                    round(counts["correct"] / counts["total"], 4)
                    if counts["total"] else 0
                )

        if neg_total:
            summary["negative/n"]            = neg_total
            summary["negative/refusal_rate"] = round(neg_refused / neg_total, 4)

        # MLflow
        mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float))})
        mlflow.log_params({
            "embed_model":    cfg.embedding.model,
            "rerank_model":   cfg.reranker.model,
            "llm_model":      cfg.llm.model,
            "top_k_retrieve": cfg.reranker.top_k_retrieve,
            "top_k_return":   cfg.reranker.top_k_return,
            "eval_records":   len(results),
        })

        # Save eval_results.jsonl
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            for r in results:
                json.dump(r, f, ensure_ascii=False)
                f.write("\n")
        mlflow.log_artifact(str(RESULTS_PATH))

        # Save eval_metrics.json (DVC metrics file)
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(str(METRICS_PATH))
        log.info(f"Metrics saved -> {METRICS_PATH}")

        # Print summary
        width = 60
        print("\n" + "=" * width)
        print("  EVAL SUMMARY")
        print("=" * width)
        for k, v in summary.items():
            bar = ""
            if isinstance(v, float) and 0.0 <= v <= 1.0:
                filled = int(v * 20)
                bar = f"  [{'#' * filled}{'.' * (20 - filled)}]"
            print(f"  {k:<42} {str(v):<6}{bar}")
        print("=" * width)
        print(f"  Full results  -> {RESULTS_PATH}")
        print(f"  DVC metrics   -> {METRICS_PATH}")
        print(f"  MLflow UI     -> mlflow ui")
        print("=" * width + "\n")


if __name__ == "__main__":
    main()