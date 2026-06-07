#src/rag/query_thesis.py
"""
Interactive query script for the corpus-aware RAG index.

The filename is kept as query_thesis.py for compatibility with the current project,
but the implementation is corpus-aware and does not hardcode thesis-specific entities.

Main improvements in this version:
1. More readable demo output.
2. Evidence table with page / section / chunk information.
3. Clearer unsupported-claim behavior.
4. Optional display of rewritten retrieval query.
"""

from __future__ import annotations

import argparse
import os
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


MAX_SOURCE_CHARS_FOR_PROMPT = 1200
MAX_EVIDENCE_PREVIEW_CHARS = 260


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

    return "\n".join(line for line in lines if line.strip())


def rewrite_query_for_retrieval(
    llm: OpenAI,
    user_query: str,
    corpus_profile: Dict[str, Any] | None = None,
) -> str:
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
7. For unsupported-claim questions, add terms such as evidence, supported, not supported, limitation, scope, generalization.
8. If the query is in Chinese, keep important Chinese terms and add helpful English equivalents.
9. Return only the rewritten retrieval query, no explanation.

User query:
{user_query}

Rewritten retrieval query:
""".strip()

    response = llm.complete(prompt)
    rewritten = str(response).strip()
    if not rewritten:
        return user_query
    return f"{user_query}\n{rewritten}"


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def compact_preview(text: str, max_chars: int = MAX_EVIDENCE_PREVIEW_CHARS) -> str:
    text = clean_text(text).replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def build_source_blocks(nodes: List[NodeWithScore]) -> List[str]:
    source_blocks = []
    for i, node_with_score in enumerate(nodes, start=1):
        node = node_with_score.node
        metadata = node.metadata or {}
        text = clean_text(node.get_content())[:MAX_SOURCE_CHARS_FOR_PROMPT]
        source_blocks.append(
            "\n".join(
                [
                    f"[Source {i}]",
                    f"score={node_with_score.score:.4f}",
                    f"page={metadata.get('page')}",
                    f"section={metadata.get('section_type')}",
                    f"section_title={metadata.get('section_title')}",
                    f"chunk_id={metadata.get('chunk_id')}",
                    "content:",
                    text,
                ]
            )
        )
    return source_blocks


def generate_answer(llm: OpenAI, question: str, nodes: List[NodeWithScore]) -> str:
    source_text = "\n\n".join(build_source_blocks(nodes))
    prompt = f"""
You are a source-grounded QA assistant for a RAG demo.
Answer the user question using only the retrieved sources.

Grounding rules:
1. Do not use outside knowledge.
2. Cite the retrieved sources in natural language, for example "Source 2" or "Sources 1 and 4".
3. If the sources do not support a claim, say clearly: "The retrieved sources do not support this claim."
4. For questions asking whether the document proves, reports, compares, evaluates, or demonstrates something, answer "yes" only if the sources explicitly support it.
5. If the evidence is partial, say what is supported and what is not supported.
6. Do not turn absence of evidence into a positive claim.
7. Be concise, but include enough explanation to show why the answer is grounded.

Question:
{question}

Retrieved sources:
{source_text}

Answer:
""".strip()
    return str(llm.complete(prompt)).strip()


def print_answer(answer: str) -> None:
    print("\nAnswer")
    print("------")
    print(textwrap.fill(answer, width=100, replace_whitespace=False))


def print_evidence(nodes: List[NodeWithScore]) -> None:
    print("\nEvidence")
    print("--------")
    for i, node_with_score in enumerate(nodes, start=1):
        node = node_with_score.node
        metadata = node.metadata or {}
        page = metadata.get("page")
        section = metadata.get("section_type")
        section_title = metadata.get("section_title")
        chunk_id = metadata.get("chunk_id")
        score = node_with_score.score
        preview = compact_preview(node.get_content())

        print(f"[{i}] page={page} | section={section} | chunk_id={chunk_id} | score={score:.4f}")
        if section_title:
            print(f"    title: {section_title}")
        print(f"    preview: {preview}")


def print_demo_summary(question: str, retrieval_query: str, show_rewritten_query: bool) -> None:
    print("\nQuestion")
    print("--------")
    print(question)

    if show_rewritten_query:
        print("\nRetrieval Query")
        print("---------------")
        print(retrieval_query)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*", help="Question to ask the RAG system.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--corpus_profile", default=None)
    parser.add_argument("--show_rewritten_query", action="store_true")
    parser.add_argument("--top_k", type=int, default=None)
    args = parser.parse_args()

    question = " ".join(args.question).strip()
    if not question:
        raise ValueError(
            "Please provide a question, for example: "
            "python src/rag/query_thesis.py \"What is the main contribution?\""
        )

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    cfg = load_config(args.config)
    profile_path = args.corpus_profile or cfg.get("paths", {}).get("corpus_profile", "configs/corpus_profile.yaml")
    corpus_profile = load_corpus_profile(profile_path)

    llm = OpenAI(model=cfg.get("llm", {}).get("model", "gpt-4o-mini"), temperature=0)
    embed_model = OpenAIEmbedding(model=cfg.get("embedding", {}).get("model", "text-embedding-3-small"))
    Settings.llm = llm
    Settings.embed_model = embed_model

    storage_dir = cfg.get("paths", {}).get("storage_dir", "storage/thesis_index")
    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
    index = load_index_from_storage(storage_context)

    top_k = args.top_k if args.top_k is not None else int(cfg.get("rag", {}).get("top_k", 8))
    use_query_rewrite = bool(cfg.get("rag", {}).get("use_query_rewrite", True))
    retriever = index.as_retriever(similarity_top_k=top_k)

    retrieval_query = rewrite_query_for_retrieval(llm, question, corpus_profile) if use_query_rewrite else question
    nodes = retriever.retrieve(retrieval_query)
    answer = generate_answer(llm, question, nodes)

    print_demo_summary(question, retrieval_query, args.show_rewritten_query)
    print_answer(answer)
    print_evidence(nodes)


if __name__ == "__main__":
    main()
