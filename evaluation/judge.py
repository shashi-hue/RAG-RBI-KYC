import logging
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

log = logging.getLogger(__name__)

_COMBINED_PROMPT = """You are evaluating a RAG system response on two criteria.

Question:
{question}

Reference answer (ground truth):
{reference}

Generated answer:
{generated}

Score BOTH metrics from 0.0 to 1.0:

FAITHFULNESS — does the generated answer match the reference ground truth?
  1.0 = fully faithful, all key facts present, no contradictions
  0.5 = partially faithful, some facts missing or slightly off
  0.0 = contradicts ground truth or completely wrong

ANSWER_RELEVANCE — does the generated answer actually address the question asked?
  1.0 = directly and completely addresses the question
  0.5 = partially addresses it, significant gaps
  0.0 = off-topic or completely ignores the question

Reply with EXACTLY this format, nothing else:
FAITHFULNESS: <float>
ANSWER_RELEVANCE: <float>"""


def score_answer(
    question: str,
    reference: str,
    generated: str,
    llm: ChatGroq,
    max_chars: int = 800,
) -> dict[str, float]:
    """
    Single LLM call → returns both faithfulness and answer_relevance.
    Returns -1.0 sentinels on failure.
    """
    try:
        prompt = _COMBINED_PROMPT.format(
            question=question[:max_chars],
            reference=reference[:max_chars],
            generated=generated[:max_chars],
        )
        result  = llm.invoke([HumanMessage(content=prompt)])
        content = result.content.strip()

        scores = {"faithfulness": -1.0, "answer_relevance": -1.0}
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("FAITHFULNESS:"):
                scores["faithfulness"] = round(
                    max(0.0, min(1.0, float(line.split(":")[1].strip()))), 4
                )
            elif line.startswith("ANSWER_RELEVANCE:"):
                scores["answer_relevance"] = round(
                    max(0.0, min(1.0, float(line.split(":")[1].strip()))), 4
                )
        return scores

    except Exception as e:
        log.warning(f"Answer scoring failed: {e}")
        return {"faithfulness": -1.0, "answer_relevance": -1.0}
