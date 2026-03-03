import os
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
from evaluation.judge import score_answer


# ---------------- CONFIG ---------------- #

EVAL_PATH    = Path("data/eval/eval_dataset.jsonl")
RESULTS_PATH = Path("data/eval/eval_results.jsonl")
METRICS_PATH = Path("data/eval/eval_metrics.json")
TOKEN_PATH   = Path("data/eval/token_usage.json")

GROQ_TPD_LIMIT  = int(os.getenv("GROQ_TPD_LIMIT", "100000"))
GROQ_TPD_BUFFER = int(os.getenv("GROQ_TPD_BUFFER", "2000"))

LLM_SLEEP_SECONDS = float(os.getenv("LLM_SLEEP_SECONDS", "3.0"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------- UTIL ---------------- #

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / 4))


def load_results():
    results = []
    processed = set()
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    results.append(r)
                    processed.add(r.get("id"))
    return results, processed


def save_results(results):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        for r in results:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")
        f.flush()


def load_token_state():
    if TOKEN_PATH.exists():
        try:
            return json.loads(TOKEN_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"total_used": 0}


def save_token_state(state):
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def remaining_tokens(token_state):
    return GROQ_TPD_LIMIT - int(token_state.get("total_used", 0))


def pause_run(results, token_state):
    save_results(results)
    save_token_state(token_state)
    log.warning(
        f"TPD buffer reached. Used={token_state['total_used']} "
        f"Remaining={remaining_tokens(token_state)}"
    )
    raise SystemExit(0)

def compute_summary_from_results(results):
    ret_hr, ret_mrr, ret_r5, ret_p5 = [], [], [], []
    ret_faith, ret_rel = [], []
    intent_results = defaultdict(lambda: {"correct": 0, "total": 0})
    neg_refused = neg_total = 0

    for r in results:
        typ = r.get("type")

        if typ in {"faq", "deep", "multihop"}:
            ret_hr.append(r.get("hit_rate", 0))
            ret_mrr.append(r.get("mrr", 0))
            ret_r5.append(r.get("recall_at_5", 0))
            ret_p5.append(r.get("precision_at_5", 0))

            if r.get("faithfulness", -1) >= 0:
                ret_faith.append(r["faithfulness"])
            if r.get("answer_relevance", -1) >= 0:
                ret_rel.append(r["answer_relevance"])

        elif typ == "intent":
            expected = r.get("expected_intent")
            if expected:
                intent_results[expected]["total"] += 1
                intent_results[expected]["correct"] += int(r.get("correct", False))

        elif typ == "negative":
            neg_total += 1
            neg_refused += int(r.get("refused", False))

    summary = {}

    if ret_hr:
        summary.update({
            "retrieval/n": len(ret_hr),
            "retrieval/hit_rate": safe_mean(ret_hr),
            "retrieval/mrr": safe_mean(ret_mrr),
            "retrieval/recall_at_5": safe_mean(ret_r5),
            "retrieval/precision_at_5": safe_mean(ret_p5),
            "retrieval/faithfulness": safe_mean(ret_faith),
            "retrieval/answer_relevance": safe_mean(ret_rel),
        })

    if intent_results:
        total = sum(v["total"] for v in intent_results.values())
        correct = sum(v["correct"] for v in intent_results.values())
        summary["intent/n"] = total
        summary["intent/accuracy_overall"] = round(correct / total, 4)

    if neg_total:
        summary["negative/n"] = neg_total
        summary["negative/refusal_rate"] = round(neg_refused / neg_total, 4)

    return summary


# ---------------- MAIN ---------------- #

def main():
    if not EVAL_PATH.exists():
        raise FileNotFoundError("Eval dataset not found.")

    cfg = get_cfg()
    chain = get_chain()
    retriever = KYCRetriever(cfg)

    dataset = [
        json.loads(line)
        for line in EVAL_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    results, processed_ids = load_results()
    token_state = load_token_state()

    log.info(f"Resuming with {len(processed_ids)} completed items.")
    log.info(f"Token state: used={token_state['total_used']} / {GROQ_TPD_LIMIT}")

    # ---------- AUTO-FINISH PARTIALS ---------- #

    for r in results:
        if r.get("note") and "judge pending" in r["note"]:
            log.info(f"Completing partial item: {r['id']}")

            scores = score_answer(
                question=r["question"],
                reference=r.get("reference_answer", ""),
                generated=r["answer"],
                llm=chain.llm,
            )

            r["faithfulness"] = scores.get("faithfulness", -1)
            r["answer_relevance"] = scores.get("answer_relevance", -1)
            r["judge_tokens"] = estimate_tokens(json.dumps(scores))
            r.pop("note", None)

            token_state["total_used"] += r["judge_tokens"]
            save_results(results)
            save_token_state(token_state)

            log.info(f"Partial completed: {r['id']}")

    # ---------- METRIC ACCUMULATORS ---------- #

    ret_hr, ret_mrr, ret_r5, ret_p5 = [], [], [], []
    ret_faith, ret_rel = [], []
    intent_results = defaultdict(lambda: {"correct": 0, "total": 0})
    neg_refused = neg_total = 0

    mlflow.set_experiment("kyc-rag-eval")

    with mlflow.start_run(run_name=f"eval_{int(time.time())}"):

        for idx, item in enumerate(dataset, start=1):

            item_id = item["id"]
            if item_id in processed_ids:
                continue

            q = item["question"]
            typ = item.get("type")

            log.info(f"[{idx}/{len(dataset)}] Processing {item_id} ({typ})")

            if remaining_tokens(token_state) <= GROQ_TPD_BUFFER:
                pause_run(results, token_state)

            t0 = time.time()

            # ---------- RETRIEVAL TYPES ---------- #

            if typ in {"faq", "deep", "multihop"}:

                expected = item.get("expected_chunk_ids", [])

                chunks = retriever.retrieve_active(q)
                retrieved_ids = [
                    c.payload.get("chunk_id", "") for c in chunks
                ]

                response = chain.query(q)

                gen_tokens = (
                    response.llm_usage.get("total_tokens")
                    if hasattr(response, "llm_usage")
                    and response.llm_usage
                    else estimate_tokens(response.answer)
                )

                token_state["total_used"] += int(gen_tokens)

                if remaining_tokens(token_state) <= GROQ_TPD_BUFFER:
                    results.append({
                        "id": item_id,
                        "type": typ,
                        "question": q,
                        "note": "partial - generation done, judge pending (paused due to TPD limit)",
                        "answer": response.answer,
                        "gen_tokens": gen_tokens,
                    })
                    pause_run(results, token_state)

                time.sleep(LLM_SLEEP_SECONDS)

                scores = score_answer(
                    question=q,
                    reference=item.get("reference_answer", ""),
                    generated=response.answer,
                    llm=chain.llm,
                )

                judge_tokens = estimate_tokens(json.dumps(scores))
                token_state["total_used"] += judge_tokens

                hr_ = hit_rate(expected, retrieved_ids)
                mrr_ = mrr(expected, retrieved_ids)
                r5_ = recall_at_k(expected, retrieved_ids, 5)
                p5_ = precision_at_k(expected, retrieved_ids, 5)

                faith_ = scores.get("faithfulness", -1)
                rel_ = scores.get("answer_relevance", -1)

                ret_hr.append(hr_)
                ret_mrr.append(mrr_)
                ret_r5.append(r5_)
                ret_p5.append(p5_)
                if faith_ >= 0:
                    ret_faith.append(faith_)
                if rel_ >= 0:
                    ret_rel.append(rel_)

                result = {
                    "id": item_id,
                    "type": typ,
                    "question": q,
                    "expected_chunk_ids": expected,
                    "retrieved_chunk_ids": retrieved_ids,
                    "hit_rate": hr_,
                    "mrr": mrr_,
                    "recall_at_5": r5_,
                    "precision_at_5": p5_,
                    "faithfulness": faith_,
                    "answer_relevance": rel_,
                    "answer": response.answer,
                    "reference_answer": item.get("reference_answer", ""),
                    "chunks_used": response.chunks_used,
                    "has_deleted": response.has_deleted_provisions,
                    "has_amended": response.has_amended_provisions,
                    "answer_tokens": answer_length_tokens(response.answer),
                    "elapsed_sec": round(time.time() - t0, 3),
                    "gen_tokens": gen_tokens,
                    "judge_tokens": judge_tokens,
                }

            # ---------- INTENT ---------- #

            elif typ == "intent":

                predicted, chapter = chain.router.classify(q)
                expected = item.get("expected_intent")

                correct = predicted.value == expected
                intent_results[expected]["total"] += 1
                intent_results[expected]["correct"] += int(correct)

                result = {
                    "id": item_id,
                    "type": typ,
                    "question": q,
                    "expected_intent": expected,
                    "predicted_intent": predicted.value,
                    "chapter_hint": chapter,
                    "correct": correct,
                    "elapsed_sec": round(time.time() - t0, 3),
                }

            # ---------- NEGATIVE ---------- #

            elif typ == "negative":

                response = chain.query(q)

                gen_tokens = estimate_tokens(response.answer)
                token_state["total_used"] += gen_tokens

                refused = is_refusal(response.answer)
                neg_refused += int(refused)
                neg_total += 1

                result = {
                    "id": item_id,
                    "type": typ,
                    "question": q,
                    "refused": refused,
                    "answer": response.answer,
                    "elapsed_sec": round(time.time() - t0, 3),
                    "gen_tokens": gen_tokens,
                }

            else:
                continue

            results.append(result)
            processed_ids.add(item_id)

            save_results(results)
            save_token_state(token_state)

        # ---------- SUMMARY (FULL DATASET) ---------- #

        summary = compute_summary_from_results(results)

        METRICS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        mlflow.log_metrics(summary)
        mlflow.log_params({"tokens_used": token_state["total_used"]})

        log.info("Evaluation completed successfully.")


if __name__ == "__main__":
    main()