#src/eval/evaluate_agentic_rag.py
"""
Evaluate corpus-aware Agentic RAG on a JSONL evaluation set.

This script runs:
question -> query rewrite -> retrieval -> evidence check -> retry if needed
-> neighbor context expansion -> answer generation -> citation audit -> revision.

It can optionally run the generic AgenticJudge.
By default, AgenticJudge uses online-style judging and does not use gold hints.
Pass --use_gold_hints only for offline benchmark analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agentic.agentic_rag import AgenticRAG, load_corpus_profile  # noqa: E402
from src.eval.agentic_judge import AgenticJudge  # noqa: E402


def load_config(config_path: str = "configs/baseline.yaml") -> Dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_eval_set(eval_path: Path) -> List[Dict[str, Any]]:
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_path}")

    items = []
    with eval_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no}: {e}") from e

    if not items:
        raise RuntimeError(f"No eval items loaded from {eval_path}")
    return items


def average(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def normalize_expected_section_types(item: Dict[str, Any]) -> List[str]:
    value = item.get("expected_section_type", [])
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    return []


def is_unsupported_case(item: Dict[str, Any]) -> bool:
    category = str(item.get("category", "")).lower()
    expected_behavior = str(item.get("expected_behavior", "")).lower()
    return (
        category in {"unsupported", "citation_audit"}
        or "unsupported" in expected_behavior
        or "refuse" in expected_behavior
        or "reject" in expected_behavior
        or "avoid_overclaiming" in expected_behavior
    )


def answer_is_cautious(answer: str) -> bool:
    markers = [
        "not supported", "does not support", "does not explicitly support", "do not support",
        "not report", "does not report", "do not explicitly confirm", "does not explicitly confirm",
        "not compare", "does not compare", "does not explicitly compare",
        "not prove", "does not prove", "does not explicitly prove", "no evidence",
        "cannot conclude", "cannot be confirmed", "insufficient evidence", "unsupported",
        "不支持", "没有证据", "未报告", "没有报告", "未比较", "没有比较", "不能证明", "无法证明",
    ]
    lower = answer.lower()
    return any(marker in lower for marker in markers)


def page_overlap(retrieved_page: Any, retrieved_page_end: Any, expected_pages: List[int]) -> bool:
    if not expected_pages:
        return False
    try:
        start = int(retrieved_page)
    except Exception:
        return False
    try:
        end = int(retrieved_page_end) if retrieved_page_end is not None else start
    except Exception:
        end = start
    retrieved_pages = set(range(start, end + 1))
    return bool(retrieved_pages.intersection(set(int(p) for p in expected_pages)))


def keyword_recall(retrieved: List[Dict[str, Any]], keywords: List[str], k: int) -> float:
    if not keywords:
        return 0.0
    text = "\n".join(r.get("text", "") for r in retrieved[:k]).lower()
    hits = sum(1 for kw in keywords if str(kw).lower() in text)
    return hits / len(keywords)


def compute_retrieval_metrics(item: Dict[str, Any], retrieved: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    expected_pages = [int(p) for p in item.get("expected_pages", [])]
    expected_sections = normalize_expected_section_types(item)
    answer_keywords = [str(x) for x in item.get("answer_keywords", [])]

    metrics: Dict[str, Any] = {}
    for k in [1, 3, 5, top_k]:
        top_items = retrieved[:k]
        metrics[f"page_hit@{k}"] = any(
            page_overlap(
                r.get("metadata", {}).get("page"),
                r.get("metadata", {}).get("page_end"),
                expected_pages,
            )
            for r in top_items
        )
        metrics[f"section_hit@{k}"] = bool(expected_sections) and any(
            str(r.get("metadata", {}).get("section_type")) in expected_sections
            for r in top_items
        )
        metrics[f"keyword_recall@{k}"] = keyword_recall(retrieved, answer_keywords, k)
    return metrics


def write_run_notes(output_dir: Path, summary: Dict[str, Any], run_name: str, eval_file: str) -> None:
    notes = f"""# Agentic RAG Run Notes

## Run

- run_name: `{run_name}`
- eval_file: `{eval_file}`
- corpus_profile: `{summary.get('corpus_profile')}`
- use_gold_hints: `{summary.get('use_gold_hints')}`

## Summary

```json
{json.dumps(summary, ensure_ascii=False, indent=2)}
```

## Notes

This is the corpus-aware version. Document-specific entities are read from `configs/corpus_profile.yaml` instead of being hardcoded in prompts or deterministic checks.
"""
    (output_dir / "RUN_NOTES.md").write_text(notes, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--eval_file", default=None)
    parser.add_argument("--output_dir", default="outputs/agentic_rag")
    parser.add_argument("--run_name", default="agentic_eval")
    parser.add_argument("--corpus_profile", default=None)
    parser.add_argument("--max_retry", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--no_advanced_judge", action="store_true")
    parser.add_argument("--use_gold_hints", action="store_true", help="Use eval-set gold hints only for offline benchmark analysis.")
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    cfg = load_config(args.config)
    eval_path = Path(args.eval_file or cfg.get("paths", {}).get("eval_file", "data/eval/eval_set.jsonl"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_path = args.corpus_profile or cfg.get("paths", {}).get("corpus_profile", "configs/corpus_profile.yaml")
    corpus_profile = load_corpus_profile(profile_path)

    llm = OpenAI(model=cfg.get("llm", {}).get("model", "gpt-4o-mini"), temperature=0)
    embed_model = OpenAIEmbedding(model=cfg.get("embedding", {}).get("model", "text-embedding-3-small"))
    Settings.llm = llm
    Settings.embed_model = embed_model

    storage_dir = cfg.get("paths", {}).get("storage_dir", "storage/thesis_index")
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    index = load_index_from_storage(storage_context)

    top_k = args.top_k or int(cfg.get("rag", {}).get("top_k", 8))
    max_retry = args.max_retry
    if max_retry is None:
        max_retry = int(cfg.get("agent", {}).get("max_retry", 1))

    rag = AgenticRAG(
        llm=llm,
        index=index,
        storage_context=storage_context,
        top_k=top_k,
        max_retry=max_retry,
        max_sources_for_answer=int(cfg.get("agent", {}).get("max_sources_for_answer", 8)),
        max_sources_for_judge=int(cfg.get("agent", {}).get("max_sources_for_judge", 6)),
        corpus_profile=corpus_profile,
    )

    judge = None if args.no_advanced_judge else AgenticJudge(llm=llm, use_gold_hints=args.use_gold_hints)

    items = load_eval_set(eval_path)
    all_results: List[Dict[str, Any]] = []

    result_path = output_dir / f"{args.run_name}_results.jsonl"
    summary_path = output_dir / f"{args.run_name}_summary.json"

    for item in items:
        qid = item.get("id", "unknown")
        question = item.get("question", "")
        print(f"\nRunning {qid}: {question}")

        rag_result = rag.answer(question)
        retrieval_metrics = compute_retrieval_metrics(item, rag_result.get("final_sources", []), top_k)

        advanced_judge = None
        if judge is not None:
            advanced_judge = judge.evaluate(
                item=item,
                answer=rag_result.get("final_answer", ""),
                sources=rag_result.get("final_sources", []),
            )

        unsupported_success = None
        if is_unsupported_case(item):
            unsupported_success = answer_is_cautious(rag_result.get("final_answer", ""))

        result = {
            "id": qid,
            "category": item.get("category"),
            "question": question,
            "expected_behavior": item.get("expected_behavior"),
            "retrieval_metrics": retrieval_metrics,
            "unsupported_case": is_unsupported_case(item),
            "unsupported_refusal_success": unsupported_success,
            "rag_result": rag_result,
            "advanced_judge": advanced_judge,
        }
        all_results.append(result)

        final_audit = rag_result.get("final_audit", {})
        print(
            f"{qid}: retry={rag_result.get('retry_count')}, revised={rag_result.get('revised')}, "
            f"overall={final_audit.get('overall_score')}, verdict={final_audit.get('verdict')}, "
            f"page_hit@{top_k}={retrieval_metrics.get(f'page_hit@{top_k}')}"
        )

    with result_path.open("w", encoding="utf-8") as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    final_audits = [r["rag_result"].get("final_audit", {}) for r in all_results]
    retrieval_metrics_list = [r.get("retrieval_metrics", {}) for r in all_results]
    unsupported_cases = [r for r in all_results if r.get("unsupported_case")]

    summary: Dict[str, Any] = {
        "run_name": args.run_name,
        "eval_file": str(eval_path),
        "corpus_profile": str(profile_path),
        "use_gold_hints": args.use_gold_hints,
        "num_items": len(all_results),
        "top_k": top_k,
        "max_retry": max_retry,
        "avg_correctness": average([float(a.get("correctness", 0)) for a in final_audits]),
        "avg_completeness": average([float(a.get("completeness", 0)) for a in final_audits]),
        "avg_groundedness": average([float(a.get("groundedness", 0)) for a in final_audits]),
        "avg_citation_support": average([float(a.get("citation_support", 0)) for a in final_audits]),
        "avg_hallucination_risk": average([float(a.get("hallucination_risk", 0)) for a in final_audits]),
        "avg_overall_score": average([float(a.get("overall_score", 0)) for a in final_audits]),
        "num_retrieval_retries": sum(int(r["rag_result"].get("retry_count", 0)) for r in all_results),
        "revision_rate": average([1.0 if r["rag_result"].get("revised") else 0.0 for r in all_results]),
        "neighbor_context_usage_rate": average([1.0 if r["rag_result"].get("used_neighbor_context") else 0.0 for r in all_results]),
        f"page_hit@{top_k}": average([1.0 if m.get(f"page_hit@{top_k}") else 0.0 for m in retrieval_metrics_list]),
        f"section_hit@{top_k}": average([1.0 if m.get(f"section_hit@{top_k}") else 0.0 for m in retrieval_metrics_list]),
        f"keyword_recall@{top_k}": average([float(m.get(f"keyword_recall@{top_k}", 0.0)) for m in retrieval_metrics_list]),
        "unsupported_cases": len(unsupported_cases),
        "unsupported_refusal_success_rate": average([
            1.0 if r.get("unsupported_refusal_success") else 0.0 for r in unsupported_cases
        ]) if unsupported_cases else None,
        "verdict_counts": {},
    }

    for audit in final_audits:
        verdict = str(audit.get("verdict", "unknown"))
        summary["verdict_counts"][verdict] = summary["verdict_counts"].get(verdict, 0) + 1

    if judge is not None:
        judge_scores = [r.get("advanced_judge", {}) for r in all_results if r.get("advanced_judge")]
        final_scores = [j.get("final_score", {}) for j in judge_scores]
        claim_metrics = [j.get("claim_metrics", {}) for j in judge_scores]
        summary["advanced_judge"] = {
            "enabled": True,
            "use_gold_hints": args.use_gold_hints,
            "avg_overall_score": average([float(s.get("overall_score", 0)) for s in final_scores]),
            "avg_claim_support_rate": average([float(m.get("claim_support_rate", 0.0)) for m in claim_metrics]),
            "total_unsupported_claims": sum(int(m.get("num_unsupported", 0)) for m in claim_metrics),
            "total_critical_mismatch_count": sum(int(m.get("critical_mismatch_count", 0)) for m in claim_metrics),
        }
    else:
        summary["advanced_judge"] = {"enabled": False}

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_run_notes(output_dir, summary, args.run_name, str(eval_path))

    print("\nAgentic RAG evaluation finished")
    print("-------------------------------")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved results: {result_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved notes: {output_dir / 'RUN_NOTES.md'}")


if __name__ == "__main__":
    main()

