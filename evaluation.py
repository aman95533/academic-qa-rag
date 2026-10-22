"""Repeatable evaluation helpers for retrieval and answer quality."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from generation import generate_grounded_answer, AnswerMode
from retrieval import RetrievedChunk, VectorStore

SAMPLE_DATASET_PATH = Path(__file__).parent / "data" / "eval_sample.json"


@dataclass
class EvalQuery:
    query_id: str
    question: str
    relevant_chunk_substrings: list[str]
    reference_answer: str | None = None


@dataclass
class RetrievalEvalResult:
    query_id: str
    question: str
    hit_at_k: bool
    retrieved_preview: list[str]
    latency_seconds: float


@dataclass
class AnswerEvalResult:
    query_id: str
    question: str
    generated_answer: str
    reference_answer: str | None
    insufficient_evidence: bool
    latency_seconds: float
    notes: list[str] = field(default_factory=list)


@dataclass
class EvaluationReport:
    retrieval: list[RetrievalEvalResult]
    answers: list[AnswerEvalResult]
    summary: dict


def load_sample_dataset(path: Path | None = None) -> list[EvalQuery]:
    path = path or SAMPLE_DATASET_PATH
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("queries", [])
    out: list[EvalQuery] = []
    for item in items:
        out.append(
            EvalQuery(
                query_id=str(item["query_id"]),
                question=item["question"],
                relevant_chunk_substrings=list(
                    item.get("relevant_chunk_substrings", [])
                ),
                reference_answer=item.get("reference_answer"),
            )
        )
    return out


def retrieval_hit_at_k(
    retrieved: list[RetrievedChunk], expected_substrings: list[str]
) -> bool:
    if not expected_substrings:
        return False
    corpus = " ".join(c.text.lower() for c in retrieved)
    return any(sub.lower() in corpus for sub in expected_substrings)


def evaluate_retrieval(
    store: VectorStore,
    queries: list[EvalQuery],
    top_k: int = 5,
) -> list[RetrievalEvalResult]:
    results: list[RetrievalEvalResult] = []
    for q in queries:
        start = time.perf_counter()
        try:
            chunks = store.retrieve(q.question, top_k=top_k)
        except Exception:
            chunks = []
        latency = time.perf_counter() - start
        results.append(
            RetrievalEvalResult(
                query_id=q.query_id,
                question=q.question,
                hit_at_k=retrieval_hit_at_k(chunks, q.relevant_chunk_substrings),
                retrieved_preview=[c.text[:120] for c in chunks[:3]],
                latency_seconds=latency,
            )
        )
    return results


def evaluate_answers(
    store: VectorStore,
    queries: list[EvalQuery],
    top_k: int = 5,
) -> list[AnswerEvalResult]:
    results: list[AnswerEvalResult] = []
    for q in queries:
        notes: list[str] = []
        start = time.perf_counter()
        try:
            chunks = store.retrieve(q.question, top_k=top_k)
            gen = generate_grounded_answer(q.question, chunks, mode=AnswerMode.SHORT)
        except Exception as exc:
            chunks = []
            gen = None
            notes.append(str(exc))
        latency = time.perf_counter() - start

        if gen is None:
            results.append(
                AnswerEvalResult(
                    query_id=q.query_id,
                    question=q.question,
                    generated_answer="",
                    reference_answer=q.reference_answer,
                    insufficient_evidence=True,
                    latency_seconds=latency,
                    notes=notes,
                )
            )
            continue

        if q.reference_answer:
            notes.append(
                "Correctness vs reference requires human judgment (automated check not implemented)."
            )
        notes.append(
            "Faithfulness to evidence requires human review of retrieved passages."
        )

        results.append(
            AnswerEvalResult(
                query_id=q.query_id,
                question=q.question,
                generated_answer=gen.answer,
                reference_answer=q.reference_answer,
                insufficient_evidence=gen.insufficient_evidence,
                latency_seconds=latency,
                notes=notes,
            )
        )
    return results


def build_report(
    retrieval_results: list[RetrievalEvalResult],
    answer_results: list[AnswerEvalResult],
) -> EvaluationReport:
    n = len(retrieval_results)
    hits = sum(1 for r in retrieval_results if r.hit_at_k)
    avg_ret_latency = (
        sum(r.latency_seconds for r in retrieval_results) / n if n else 0.0
    )
    avg_ans_latency = (
        sum(a.latency_seconds for a in answer_results) / len(answer_results)
        if answer_results
        else 0.0
    )
    summary = {
        "retrieval_queries": n,
        "retrieval_hit_at_k": hits,
        "retrieval_hit_rate": (hits / n) if n else None,
        "avg_retrieval_latency_seconds": round(avg_ret_latency, 4),
        "avg_answer_latency_seconds": round(avg_ans_latency, 4),
        "automated_metrics": [
            "retrieval_hit_at_k (substring match in top-k)",
            "latency (retrieval and end-to-end answer)",
        ],
        "human_judgment_metrics": [
            "answer_faithfulness_to_evidence",
            "answer_correctness_vs_reference",
            "answer_relevance",
        ],
    }
    return EvaluationReport(
        retrieval=retrieval_results, answers=answer_results, summary=summary
    )


def run_evaluation(
    store: VectorStore,
    dataset_path: Path | None = None,
    top_k: int = 5,
) -> EvaluationReport:
    queries = load_sample_dataset(dataset_path)
    if not queries:
        return EvaluationReport(
            retrieval=[],
            answers=[],
            summary={
                "error": f"No evaluation queries found at {dataset_path or SAMPLE_DATASET_PATH}"
            },
        )
    retr = evaluate_retrieval(store, queries, top_k=top_k)
    ans = evaluate_answers(store, queries, top_k=top_k)
    return build_report(retr, ans)


def main() -> None:
    """CLI entry: python -m evaluation"""
    store = VectorStore()
    report = run_evaluation(store)
    print(json.dumps(report.summary, indent=2))
    for r in report.retrieval:
        print(f"[{r.query_id}] hit={r.hit_at_k} q={r.question[:60]}...")


if __name__ == "__main__":
    main()
