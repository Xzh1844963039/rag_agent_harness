#src/eval/evaluate_answer.py
"""
Answer-level evaluation for the Agentic RAG harness.

This script implements a lightweight LLM-as-a-Judge evaluator for RAG answers.
It follows common RAG evaluation ideas:

1. Answer correctness:
   whether the generated answer correctly addresses the question.

2. Completeness:
   whether the answer covers the major expected points.

3. Groundedness / faithfulness:
   whether the answer is supported by the retrieved source chunks.

4. Citation support:
   whether the retrieved evidence is sufficient to support the main claims.

5. Hallucination risk:
   whether the answer contains unsupported claims, invented numbers,
   wrong model names, wrong benchmark names, or overconfident conclusions.

The scores are not treated as absolute ground truth. They are used as an
automatic evaluation signal for comparing RAG variants and identifying
failure cases before manual inspection.
"""
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


def load_config(config_path: str = "configs/baseline.yaml") -> dict:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def rewrite_query_for_retrieval(llm: OpenAI, user_query: str) -> str:
    prompt = f"""
You are a query rewriting module for a RAG system over an undergraduate thesis.

The thesis is about:
- Student-oriented Chain-of-Thought optimization
- Teacher-Student-Controller framework
- local CoT repair
- key-step localization
- student model learning
- Qwen2.5-Math-1.5B and Qwen2.5-Math-7B
- QLoRA+DoRA supervised fine-tuning
- math500 strict evaluation
- experimental results
- limitations and future work

Rewrite the user query into a retrieval query that is easier to match against thesis chunks.

Strict rules:
1. Do not answer the question.
2. Preserve the original intent.
3. Do not add unrelated topics.
4. Add only closely related thesis-specific keywords.
5. If the user asks for numerical results, include likely table/result keywords such as:
   Table 2, Main Results, math500 strict, Original CoTs, Fixed CoTs,
   Qwen2.5-Math-1.5B, Qwen2.5-Math-7B.
6. If the user asks about limitations or future work, include:
   remaining limited, future work, generalization, evaluation, larger datasets, conclusion.
7. If the query is in Chinese, keep important Chinese terms and add relevant English thesis terms.
8. Return only the rewritten retrieval query, no explanation.

User query:
{user_query}

Rewritten retrieval query:
""".strip()

    response = llm.complete(prompt)
    rewritten = str(response).strip()

    return f"{user_query}\n{rewritten}"


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def source_nodes_to_records(source_nodes: List[Any], max_preview_chars: int = 900) -> List[Dict[str, Any]]:
    records = []

    for rank, source_node in enumerate(source_nodes, start=1):
        node = source_node.node
        metadata = node.metadata or {}
        text = node.get_content()

        records.append(
            {
                "rank": rank,
                "score": safe_float(source_node.score),
                "metadata": {
                    "source": metadata.get("source"),
                    "page": metadata.get("page"),
                    "page_end": metadata.get("page_end"),
                    "section_type": metadata.get("section_type"),
                    "section_title": metadata.get("section_title"),
                    "unit_type": metadata.get("unit_type"),
                    "chunk_id": metadata.get("chunk_id"),
                    "page_chunk_id": metadata.get("page_chunk_id"),
                    "previous_chunk_id": metadata.get("previous_chunk_id"),
                    "next_chunk_id": metadata.get("next_chunk_id"),
                    "parser": metadata.get("parser"),
                    "char_len": metadata.get("char_len"),
                    "chunking": metadata.get("chunking"),
                },
                "text": text,
                "preview": text.replace("\n", " ")[:max_preview_chars],
            }
        )

    return records


def format_sources_for_judge(sources: List[Dict[str, Any]], max_sources: int = 6, max_chars_each: int = 1300) -> str:
    parts = []

    for src in sources[:max_sources]:
        meta = src.get("metadata", {})
        text = src.get("text", "")
        text = text.replace("\n", " ").strip()

        if len(text) > max_chars_each:
            text = text[:max_chars_each] + "..."

        parts.append(
            f"[Source {src.get('rank')}]\n"
            f"page={meta.get('page')}, page_end={meta.get('page_end')}, "
            f"section_type={meta.get('section_type')}, "
            f"section_title={meta.get('section_title')}, "
            f"chunk_id={meta.get('chunk_id')}, "
            f"unit_type={meta.get('unit_type')}\n"
            f"text={text}"
        )

    return "\n\n".join(parts)


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Try to parse a JSON object from an LLM response.
    The judge prompt asks for pure JSON, but this function is defensive.
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        raise ValueError(f"Could not find JSON object in judge response:\n{text}")

    candidate = match.group(0)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse JSON object from judge response:\n{candidate}") from e


def clamp_score(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        return default

    if v < 0:
        return 0.0
    if v > 5:
        return 5.0

    return v


def judge_answer(
    llm: OpenAI,
    item: Dict[str, Any],
    answer: str,
    sources: List[Dict[str, Any]],
) -> Dict[str, Any]:
    question = item.get("question", "")
    category = item.get("category", "")
    expected_pages = item.get("expected_pages", [])
    expected_sections = item.get("expected_section_type", [])
    answer_keywords = item.get("answer_keywords", [])

    source_text = format_sources_for_judge(sources)

    prompt = f"""
You are an evaluator for a source-grounded RAG system.

Your task is to judge whether the answer is correct and supported by the retrieved sources.

Question:
{question}

Question category:
{category}

Expected evidence hints:
- expected_pages: {expected_pages}
- expected_section_type: {expected_sections}
- answer_keywords: {answer_keywords}

RAG answer:
{answer}

Retrieved sources:
{source_text}

Evaluation rules:
1. Judge based only on the retrieved sources and the expected evidence hints.
2. Do not require the answer to use the exact same wording as the sources.
3. Reward answers that are faithful, specific, and grounded.
4. Penalize unsupported claims, invented numbers, wrong model names, wrong benchmark names, or overconfident conclusions.
5. If the answer is generally correct but misses some important details, lower completeness but not necessarily groundedness.
6. If the retrieved sources are relevant but the answer fails to use them, lower correctness and completeness.
7. If the answer contains claims that cannot be verified from the retrieved sources, list them as unsupported_claims.
8. Return valid JSON only. Do not include markdown.

Return this JSON object:
{{
  "correctness": 0-5,
  "completeness": 0-5,
  "groundedness": 0-5,
  "citation_support": 0-5,
  "hallucination_risk": 0-5,
  "overall_score": 0-5,
  "verdict": "excellent" | "good" | "partial" | "poor",
  "reason": "brief explanation",
  "supported_claims": ["claim 1", "claim 2"],
  "unsupported_claims": ["claim 1", "claim 2"],
  "missing_points": ["missing point 1", "missing point 2"]
}}
""".strip()

    response = llm.complete(prompt)
    raw = str(response).strip()

    try:
        parsed = extract_json_object(raw)
    except Exception as e:
        parsed = {
            "correctness": 0,
            "completeness": 0,
            "groundedness": 0,
            "citation_support": 0,
            "hallucination_risk": 5,
            "overall_score": 0,
            "verdict": "poor",
            "reason": f"Judge JSON parsing failed: {e}",
            "supported_claims": [],
            "unsupported_claims": [],
            "missing_points": [],
            "raw_judge_response": raw,
        }

    parsed["correctness"] = clamp_score(parsed.get("correctness"))
    parsed["completeness"] = clamp_score(parsed.get("completeness"))
    parsed["groundedness"] = clamp_score(parsed.get("groundedness"))
    parsed["citation_support"] = clamp_score(parsed.get("citation_support"))
    parsed["hallucination_risk"] = clamp_score(parsed.get("hallucination_risk"), default=5.0)
    parsed["overall_score"] = clamp_score(parsed.get("overall_score"))

    for key in ["supported_claims", "unsupported_claims", "missing_points"]:
        if key not in parsed or not isinstance(parsed[key], list):
            parsed[key] = []

    if "verdict" not in parsed:
        parsed["verdict"] = "partial"

    if "reason" not in parsed:
        parsed["reason"] = ""

    return parsed


def compute_retrieval_snapshot(
    sources: List[Dict[str, Any]],
    expected_pages: List[int],
    expected_sections: List[str],
    answer_keywords: List[str],
    k: int,
) -> Dict[str, Any]:
    top = sources[:k]

    retrieved_pages = []
    retrieved_sections = []
    retrieved_text = []

    for src in top:
        meta = src.get("metadata", {})

        page = meta.get("page")
        page_end = meta.get("page_end", page)

        if page is not None:
            try:
                p1 = int(page)
                p2 = int(page_end) if page_end is not None else p1
                retrieved_pages.extend(list(range(p1, p2 + 1)))
            except Exception:
                pass

        section = meta.get("section_type")
        if section:
            retrieved_sections.append(str(section))

        retrieved_text.append(src.get("text", ""))

    expected_page_set = set(int(p) for p in expected_pages)
    retrieved_page_set = set(retrieved_pages)

    page_hit = bool(expected_page_set & retrieved_page_set) if expected_page_set else False
    section_hit = bool(set(expected_sections) & set(retrieved_sections)) if expected_sections else False

    joined_text = "\n".join(retrieved_text).lower()

    keyword_hits = []
    for kw in answer_keywords:
        if str(kw).lower() in joined_text:
            keyword_hits.append(kw)

    keyword_recall = len(keyword_hits) / len(answer_keywords) if answer_keywords else 0.0

    return {
        f"page_hit@{k}": page_hit,
        f"section_hit@{k}": section_hit,
        f"keyword_recall@{k}": keyword_recall,
        f"keyword_hits@{k}": keyword_hits,
    }


def average(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    load_dotenv()
    config = load_config()

    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is missing. Please check your .env file.")

    storage_dir = Path(config["paths"]["storage_dir"])
    eval_path = Path(config["paths"].get("eval_file", "data/eval/eval_set.jsonl"))
    output_dir = Path(config["paths"].get("output_dir", "outputs/baseline_run"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if not storage_dir.exists():
        raise FileNotFoundError(
            f"Index storage dir not found: {storage_dir}\n"
            "Please run: python src\\indexing\\build_index.py"
        )

    llm_model = config["llm"]["model"]
    embedding_model = config["embedding"]["model"]
    top_k = int(config["rag"].get("top_k", 8))
    use_query_rewrite = bool(config["rag"].get("use_query_rewrite", True))

    llm = OpenAI(model=llm_model, api_key=openai_api_key)
    embed_model = OpenAIEmbedding(model=embedding_model, api_key=openai_api_key)

    Settings.llm = llm
    Settings.embed_model = embed_model

    storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))
    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        response_mode="compact",
    )

    eval_items = load_eval_set(eval_path)

    result_path = output_dir / "answer_eval_results.jsonl"
    summary_path = output_dir / "answer_eval_summary.json"

    all_results = []

    print("Answer evaluation started")
    print("-------------------------")
    print(f"Eval file: {eval_path}")
    print(f"Storage dir: {storage_dir}")
    print(f"Output dir: {output_dir}")
    print(f"LLM model: {llm_model}")
    print(f"Embedding model: {embedding_model}")
    print(f"Top k: {top_k}")
    print(f"Use query rewrite: {use_query_rewrite}")

    for idx, item in enumerate(eval_items, start=1):
        qid = item["id"]
        question = item["question"]

        print(f"\n[{idx}/{len(eval_items)}] {qid}: {question}")

        retrieval_query = (
            rewrite_query_for_retrieval(llm, question)
            if use_query_rewrite
            else question
        )

        response = query_engine.query(retrieval_query)

        answer = str(response).strip()
        sources = source_nodes_to_records(response.source_nodes)

        retrieval_snapshot = {}
        for k in [1, 3, 5, top_k]:
            k = min(k, top_k)
            retrieval_snapshot.update(
                compute_retrieval_snapshot(
                    sources=sources,
                    expected_pages=item.get("expected_pages", []),
                    expected_sections=item.get("expected_section_type", []),
                    answer_keywords=item.get("answer_keywords", []),
                    k=k,
                )
            )

        judge = judge_answer(
            llm=llm,
            item=item,
            answer=answer,
            sources=sources,
        )

        result = {
            "id": qid,
            "category": item.get("category"),
            "question": question,
            "question_zh": item.get("question_zh"),
            "retrieval_query": retrieval_query,
            "answer": answer,
            "expected_pages": item.get("expected_pages", []),
            "expected_section_type": item.get("expected_section_type", []),
            "answer_keywords": item.get("answer_keywords", []),
            "retrieval_snapshot": retrieval_snapshot,
            "judge": judge,
            "sources": sources,
        }

        all_results.append(result)

        print(
            f"overall={judge.get('overall_score'):.2f}, "
            f"correctness={judge.get('correctness'):.2f}, "
            f"groundedness={judge.get('groundedness'):.2f}, "
            f"citation_support={judge.get('citation_support'):.2f}, "
            f"verdict={judge.get('verdict')}"
        )

        # 防止连续请求太快。如果你不想等，可以把这个 sleep 改成 0。
        time.sleep(0.5)

    with result_path.open("w", encoding="utf-8") as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    correctness_values = [r["judge"]["correctness"] for r in all_results]
    completeness_values = [r["judge"]["completeness"] for r in all_results]
    groundedness_values = [r["judge"]["groundedness"] for r in all_results]
    citation_values = [r["judge"]["citation_support"] for r in all_results]
    hallucination_values = [r["judge"]["hallucination_risk"] for r in all_results]
    overall_values = [r["judge"]["overall_score"] for r in all_results]

    verdict_counts: Dict[str, int] = {}
    category_scores: Dict[str, List[float]] = {}

    for result in all_results:
        verdict = result["judge"].get("verdict", "unknown")
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

        category = result.get("category", "unknown")
        category_scores.setdefault(category, []).append(result["judge"]["overall_score"])

    category_avg_scores = {
        category: average(scores)
        for category, scores in sorted(category_scores.items())
    }

    summary = {
        "num_items": len(all_results),
        "top_k": top_k,
        "use_query_rewrite": use_query_rewrite,
        "avg_correctness": average(correctness_values),
        "avg_completeness": average(completeness_values),
        "avg_groundedness": average(groundedness_values),
        "avg_citation_support": average(citation_values),
        "avg_hallucination_risk": average(hallucination_values),
        "avg_overall_score": average(overall_values),
        "verdict_counts": verdict_counts,
        "category_avg_scores": category_avg_scores,
        "result_path": str(result_path),
    }

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nAnswer evaluation finished")
    print("--------------------------")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved results: {result_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()