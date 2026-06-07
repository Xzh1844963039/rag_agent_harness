#src/eval/evaluate_retrieval.py
"""
Evaluate retrieval over a JSONL eval set.

This version keeps your page/section/keyword metrics, but removes hardcoded
thesis-specific query rewrite prompts. Query rewriting now uses corpus_profile.yaml.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: str = "configs/baseline.yaml") -> Dict[str, Any]:
    return load_yaml(config_path)


def load_corpus_profile(profile_path: str | Path | None) -> Dict[str, Any]:
    if not profile_path:
        return {}
    path = Path(profile_path)
    if not path.exists():
        return {}
    return load_yaml(path)


def format_profile(profile: Dict[str, Any]) -> str:
    corpus = profile.get("corpus", {}) or {}
    lines = [
        f"Corpus title: {corpus.get('title', '')}",
        f"Corpus description: {corpus.get('description', '')}",
        f"Corpus domain: {corpus.get('domain', '')}",
    ]
    topics = corpus.get("topics", []) or []
    if topics:
        lines.append("Topics:")
        lines.extend(f"- {x}" for x in topics)
    optional_keywords = corpus.get("optional_keywords", []) or []
    if optional_keywords:
        lines.append("Useful retrieval keywords:")
        lines.extend(f"- {x}" for x in optional_keywords)
    entity_types = corpus.get("entity_types", {}) or {}
    for entity_type, values in entity_types.items():
        if isinstance(values, list) and values:
            lines.append(f"{entity_type}: {', '.join(str(v) for v in values)}")
    return "\n".join(lines)


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


def rewrite_query_for_retrieval(llm: OpenAI, user_query: str, corpus_profile: Dict[str, Any] | None = None) -> str:
    profile_text = format_profile(corpus_profile or {})
    prompt = f"""
You are a query rewriting module for a source-grounded RAG system.

Use the corpus profile only as optional context. Do not force irrelevant profile terms into the query.

Corpus profile:
{profile_text}

Rewrite the user query into a retrieval query that is easier to match against document chunks.

Strict rules:
1. Do not answer the question.
2. Preserve the original intent.
3. Do not add unsupported assumptions.
4. Add only closely related terms that help retrieval.
5. For numerical, table, figure, model, dataset, benchmark, metric, or comparison questions, add useful generic words such as table, figure, result, metric, dataset, benchmark, model, experiment, comparison, evaluation.
6. For limitation or future-work questions, add useful terms such as limitation, future work, generalization, scope, conclusion.
7. If the query is in Chinese, keep important Chinese terms and add helpful English equivalents.
8. Return only the rewritten retrieval query, no explanation.

User query:
{user_query}

Rewritten retrieval query:
""".strip()

    response = llm.complete(prompt)
    rewritten = str(response).strip()
    if not rewritten:
        return user_query
    return f"{user_query}\n{rewritten}"


def normalize_expected_section_types(item: Dict[str, Any]) -> List[str]:
    value = item.get("expected_section_type", [])
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    return []


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


def compute_metrics(
    retrieved: List[Dict[str, Any]],
    expected_pages: List[int],
    expected_sections: List[str],
    answer_keywords: List[str],
    top_k: int,
) -> Dict[str, Any]:
    cutoffs = [1, 3, 5, top_k]
    metrics: Dict[str, Any] = {}

    for k in cutoffs:
        top_items = retrieved[:k]
        page_hit = any(
            page_overlap(
                r.get("metadata", {}).get("page"),
                r.get("metadata", {}).get("page_end"),
                expected_pages,
            )
            for r in top_items
        )
        section_hit = False
        if expected_sections:
            section_hit = any(
                str(r.get("metadata", {}).get("section_type")) in expected_sections
                for r in top_items
            )
        metrics[f"page_hit@{k}"] = page_hit
        metrics[f"section_hit@{k}"] = section_hit
        metrics[f"keyword_recall@{k}"] = keyword_recall(retrieved, answer_keywords, k)

    return metrics


def average(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--eval_file", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--corpus_profile", default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--no_query_rewrite", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    cfg = load_config(args.config)
    eval_path = Path(args.eval_file or cfg.get("paths", {}).get("eval_file", "data/eval/eval_set.jsonl"))
    output_dir = Path(args.output_dir or cfg.get("paths", {}).get("output_dir", "outputs/baseline_run"))
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
    use_query_rewrite = bool(cfg.get("rag", {}).get("use_query_rewrite", True)) and not args.no_query_rewrite
    retriever = index.as_retriever(similarity_top_k=top_k)

    items = load_eval_set(eval_path)
    all_results: List[Dict[str, Any]] = []

    result_path = output_dir / "retrieval_eval_results.jsonl"
    summary_path = output_dir / "retrieval_eval_summary.json"

    for item in items:
        qid = item.get("id", "unknown")
        question = item.get("question", "")
        retrieval_query = rewrite_query_for_retrieval(llm, question, corpus_profile) if use_query_rewrite else question
        source_nodes = retriever.retrieve(retrieval_query)

        retrieved = []
        for rank, source_node in enumerate(source_nodes, start=1):
            node = source_node.node
            metadata = node.metadata or {}
            retrieved.append(
                {
                    "rank": rank,
                    "score": float(source_node.score) if source_node.score is not None else None,
                    "metadata": metadata,
                    "text": node.get_content(),
                    "preview": node.get_content().replace("\n", " ")[:700],
                }
            )

        expected_pages = [int(p) for p in item.get("expected_pages", [])]
        expected_sections = normalize_expected_section_types(item)
        answer_keywords = [str(x) for x in item.get("answer_keywords", [])]
        metrics = compute_metrics(retrieved, expected_pages, expected_sections, answer_keywords, top_k)

        result = {
            "id": qid,
            "question": question,
            "retrieval_query": retrieval_query,
            "expected_pages": expected_pages,
            "expected_section_type": expected_sections,
            "answer_keywords": answer_keywords,
            "metrics": metrics,
            "retrieved": retrieved,
        }
        all_results.append(result)

        print(
            f"{qid}: "
            f"page_hit@{top_k}={metrics.get(f'page_hit@{top_k}')}, "
            f"section_hit@{top_k}={metrics.get(f'section_hit@{top_k}')}, "
            f"keyword_recall@{top_k}={metrics.get(f'keyword_recall@{top_k}', 0.0):.2f}"
        )

    with result_path.open("w", encoding="utf-8") as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    def avg_bool(metric_name: str) -> float:
        return average([1.0 if r["metrics"].get(metric_name) else 0.0 for r in all_results])

    def avg_float(metric_name: str) -> float:
        return average([float(r["metrics"].get(metric_name, 0.0)) for r in all_results])

    cutoffs = [1, 3, 5, top_k]
    summary: Dict[str, Any] = {
        "num_items": len(all_results),
        "top_k": top_k,
        "use_query_rewrite": use_query_rewrite,
        "corpus_profile": str(profile_path),
    }
    for k in cutoffs:
        summary[f"page_hit@{k}"] = avg_bool(f"page_hit@{k}")
        summary[f"section_hit@{k}"] = avg_bool(f"section_hit@{k}")
        summary[f"keyword_recall@{k}"] = avg_float(f"keyword_recall@{k}")

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nRetrieval evaluation finished")
    print("-----------------------------")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved results: {result_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
