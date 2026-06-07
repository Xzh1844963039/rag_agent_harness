#src/rag/query_thesis.py
"""
Interactive query script for the corpus-aware RAG index.

This file keeps the historical name query_thesis.py for compatibility, but the
implementation is no longer thesis-specific. It supports:

1. Hybrid retrieval: dense embedding retrieval + sparse BM25 retrieval + RRF.
2. Query routing: metadata / overview / innovation / method / experiment / claim / publication / general.
3. Generic front-matter metadata QA: author, title, advisor, department, program,
   date, institution, affiliation, etc. are answered from front matter instead of
   hardcoded fields.
4. Route-aware retrieval filtering: different question types prefer different
   document sections.
5. Evidence-grounded answer generation with cautious handling of unsupported or
   extra-document questions.

Design principle:
- Do not hardcode answers.
- Do not treat every question containing "这篇论文 / this thesis" as metadata.
- Do not answer external judgment questions, such as publication chance, as if
  the PDF alone can decide them.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from dotenv import load_dotenv

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


MAX_SOURCE_CHARS_FOR_PROMPT = 1200
MAX_EVIDENCE_PREVIEW_CHARS = 280

NON_CONTENT_SECTIONS = {
    "acknowledgement",
    "acknowledgements",
    "references",
    "reference",
    "toc",
    "cover",
    "commitment",
}

CONTENT_SECTIONS = {
    "abstract",
    "introduction",
    "related_work",
    "method",
    "setup",
    "results",
    "conclusion",
    "section",
}

ROUTE_SECTION_PRIORITY = {
    "metadata": {
        "cover": 0,
        "section": 1,
        "abstract": 2,
        "introduction": 3,
        "method": 4,
    },
    "overview": {
        "abstract": 0,
        "introduction": 1,
        "method": 2,
        "setup": 3,
        "results": 4,
        "conclusion": 5,
        "related_work": 6,
        "section": 7,
    },
    "innovation": {
        "abstract": 0,
        "introduction": 1,
        "related_work": 2,
        "method": 3,
        "conclusion": 4,
        "results": 5,
        "section": 6,
    },
    "publication": {
        "abstract": 0,
        "introduction": 1,
        "related_work": 2,
        "method": 3,
        "results": 4,
        "conclusion": 5,
        "section": 6,
    },
    "method": {
        "method": 0,
        "introduction": 1,
        "related_work": 2,
        "abstract": 3,
        "setup": 4,
        "section": 5,
    },
    "experiment": {
        "results": 0,
        "setup": 1,
        "method": 2,
        "abstract": 3,
        "conclusion": 4,
        "section": 5,
    },
    "claim": {
        "results": 0,
        "conclusion": 1,
        "method": 2,
        "setup": 3,
        "introduction": 4,
        "related_work": 5,
        "abstract": 6,
        "section": 7,
    },
    "general": {
        "abstract": 0,
        "introduction": 1,
        "method": 2,
        "results": 3,
        "conclusion": 4,
        "related_work": 5,
        "setup": 6,
        "section": 7,
    },
}


@dataclass
class QueryPlan:
    route: str
    retrieval_query: str
    route_reason: str
    requested_attribute: str = ""
    external_caution: bool = False


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


def is_chinese_text(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


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


def contains_any(text: str, patterns: List[str]) -> bool:
    text_lower = text.lower()
    return any(p.lower() in text_lower for p in patterns)


def is_broad_overview_query(query: str) -> bool:
    q = query.strip().lower()

    chinese_patterns = [
        "主要在讲什么",
        "主要讲什么",
        "主要内容",
        "核心内容",
        "总结一下",
        "概括一下",
        "大概讲",
        "这篇文章讲什么",
        "这篇论文讲什么",
        "这篇文章主要",
        "这篇论文主要",
    ]

    english_patterns = [
        "what is this paper about",
        "what is this paper mainly about",
        "what is this thesis about",
        "what is this thesis mainly about",
        "what is this document about",
        "what is this document mainly about",
        "what does this paper mainly discuss",
        "what does this thesis mainly discuss",
        "main idea",
        "main topic",
        "summarize the paper",
        "summarize this paper",
        "summarize the thesis",
        "summarize this thesis",
        "overview of the paper",
        "overview of this paper",
        "overview of the thesis",
        "overview of this thesis",
    ]

    return contains_any(q, chinese_patterns + english_patterns)


def looks_like_external_judgment_query(query: str) -> bool:
    external_terms = [
        "投稿",
        "会议",
        "期刊",
        "录用",
        "接收",
        "ccf",
        "acl",
        "emnlp",
        "neurips",
        "iclr",
        "icml",
        "domestic conference",
        "conference",
        "journal",
        "publish",
        "publication",
        "submission",
        "acceptance",
        "venue",
    ]
    return contains_any(query, external_terms)


def looks_like_innovation_query(query: str) -> bool:
    patterns = [
        "创新点",
        "创新性",
        "贡献",
        "主要贡献",
        "novelty",
        "innovation",
        "contribution",
        "different from",
        "相比",
        "相对于",
        "区别",
        "不同",
        "基础上提出",
        "based on",
        "extends",
        "positioning",
    ]
    return contains_any(query, patterns)


def looks_like_method_query(query: str) -> bool:
    patterns = [
        "方法",
        "框架",
        "流程",
        "机制",
        "怎么做",
        "如何实现",
        "teacher-student-controller",
        "controller",
        "student signal",
        "key-step",
        "local repair",
        "framework",
        "method",
        "pipeline",
        "algorithm",
    ]
    return contains_any(query, patterns)


def looks_like_experiment_query(query: str) -> bool:
    patterns = [
        "实验",
        "结果",
        "指标",
        "表",
        "图",
        "table",
        "figure",
        "math500",
        "gsm8k",
        "accuracy",
        "benchmark",
        "dataset",
        "1.5b",
        "7b",
        "qwen",
        "qlora",
        "dora",
        "result",
        "evaluation",
        "metric",
    ]
    return contains_any(query, patterns)


def looks_like_claim_query(query: str) -> bool:
    patterns = [
        "是否",
        "有没有",
        "能不能",
        "证明",
        "支持",
        "迁移",
        "推广",
        "所有",
        "总是",
        "一定",
        "claim",
        "prove",
        "support",
        "demonstrate",
        "compare",
        "all llms",
        "all reasoning benchmarks",
        "always",
        "not supported",
        "evidence",
    ]
    return contains_any(query, patterns)


def looks_like_metadata_query(query: str) -> bool:
    """
    Generic metadata-query detector.

    It avoids sending content questions such as "创新点是什么" or "方法是什么" to
    front-matter metadata QA. Those should go to content retrieval.
    """
    if is_broad_overview_query(query):
        return False

    if looks_like_innovation_query(query):
        return False

    if looks_like_external_judgment_query(query):
        return False

    # Method / result questions are content questions unless they explicitly ask
    # about administrative fields such as title, author, advisor, department, date.
    content_terms = [
        "研究问题",
        "方法",
        "框架",
        "实验",
        "结果",
        "结论",
        "局限",
        "贡献",
        "创新",
        "main contribution",
        "method",
        "framework",
        "experiment",
        "result",
        "conclusion",
        "limitation",
    ]

    q = query.strip().lower()

    metadata_terms = [
        "作者",
        "谁写",
        "谁完成",
        "题目",
        "标题",
        "叫什么",
        "导师",
        "指导老师",
        "指导教师",
        "院系",
        "哪个系",
        "专业",
        "学校",
        "单位",
        "机构",
        "日期",
        "时间",
        "提交",
        "学号",
        "署名",
        "author",
        "wrote",
        "title",
        "advisor",
        "supervisor",
        "department",
        "program",
        "major",
        "university",
        "institution",
        "affiliation",
        "date",
        "submitted",
        "student id",
    ]

    if not contains_any(q, metadata_terms):
        return False

    # If it also contains strong content terms, keep it content unless it contains
    # explicit administrative metadata terms.
    explicit_admin_terms = [
        "作者",
        "谁写",
        "谁完成",
        "题目",
        "标题",
        "导师",
        "指导老师",
        "指导教师",
        "院系",
        "哪个系",
        "专业",
        "学校",
        "单位",
        "日期",
        "学号",
        "author",
        "wrote",
        "title",
        "advisor",
        "supervisor",
        "department",
        "program",
        "major",
        "university",
        "institution",
        "affiliation",
        "date",
        "student id",
    ]

    if contains_any(q, content_terms) and not contains_any(q, explicit_admin_terms):
        return False

    return True


def build_profile_keywords(corpus_profile: Dict[str, Any], limit: int = 45) -> str:
    corpus = corpus_profile.get("corpus", {}) or {}

    parts: List[str] = []
    for key in ["title", "domain", "description"]:
        value = corpus.get(key)
        if value:
            parts.append(str(value))

    parts.extend(str(x) for x in (corpus.get("topics", []) or []))
    parts.extend(str(x) for x in (corpus.get("optional_keywords", []) or []))

    entity_types = corpus.get("entity_types", {}) or {}
    for values in entity_types.values():
        if isinstance(values, list):
            parts.extend(str(x) for x in values)

    seen = set()
    deduped = []
    for p in parts:
        p = p.strip()
        if not p or p.lower() in seen:
            continue
        seen.add(p.lower())
        deduped.append(p)

    return " ".join(deduped[:limit])


def infer_metadata_intent_with_llm(llm: OpenAI, question: str) -> Dict[str, str]:
    """
    Infer the requested front-matter attribute in natural language.
    This is not a fixed answer extractor; it only improves retrieval terms.
    """
    prompt = f"""
You are a query understanding module for a document QA system.

The user may be asking about document metadata or front-matter information.
Infer what metadata attribute the user asks for.

Return strict JSON only:
{{
  "is_metadata_query": true or false,
  "requested_attribute": "short natural-language attribute, e.g. author, title, advisor, department, program, submission date, affiliation, institution, student id, publication venue",
  "search_terms": "bilingual retrieval terms useful for finding that attribute in front matter"
}}

Rules:
- If the user asks for innovation, contribution, method, experiment, result, conclusion, limitation, or publication chance, set is_metadata_query=false.
- If the user asks who/what/when/where about document administrative information, set is_metadata_query=true.
- Do not answer the question.

User question:
{question}
""".strip()

    fallback = {
        "is_metadata_query": "true" if looks_like_metadata_query(question) else "false",
        "requested_attribute": "document metadata",
        "search_terms": "title author advisor supervisor department program major affiliation institution university date student id 作者 题目 标题 导师 指导老师 院系 专业 学校 单位 日期 学号",
    }

    try:
        response = str(llm.complete(prompt)).strip()
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not match:
            return fallback
        data = json.loads(match.group(0))
        return {
            "is_metadata_query": str(data.get("is_metadata_query", False)).lower(),
            "requested_attribute": str(data.get("requested_attribute", "document metadata")).strip(),
            "search_terms": str(data.get("search_terms", fallback["search_terms"])).strip(),
        }
    except Exception:
        return fallback


def classify_query(
    llm: OpenAI,
    user_query: str,
    corpus_profile: Dict[str, Any],
) -> QueryPlan:
    """
    Deterministic-first query router.

    The key fix is that innovation/contribution/publication/method questions are
    not routed to metadata just because they contain "这篇论文" or "是什么".
    """
    profile_terms = build_profile_keywords(corpus_profile)

    if looks_like_metadata_query(user_query):
        intent = infer_metadata_intent_with_llm(llm, user_query)
        requested_attribute = intent.get("requested_attribute") or "document metadata"
        search_terms = intent.get("search_terms") or ""
        retrieval_query = f"""
{user_query}
requested metadata attribute: {requested_attribute}
front matter metadata title author advisor supervisor department program major affiliation institution university date student id submission publication venue
文档元信息 标题 题目 作者 导师 指导老师 指导教师 院系 专业 学校 单位 机构 日期 提交时间 学号 署名
{search_terms}
""".strip()
        return QueryPlan(
            route="metadata",
            retrieval_query=retrieval_query,
            route_reason="metadata/front-matter question",
            requested_attribute=requested_attribute,
        )

    if looks_like_external_judgment_query(user_query):
        retrieval_query = f"""
{user_query}
innovation contribution novelty related work positioning method limitation future work discussion result experiment evaluation LLM reasoning Chain-of-Thought conference venue publication suitability
创新点 贡献 相关工作 方法 结果 局限 未来工作 讨论 投稿 会议 大模型 推理 思维链
{profile_terms}
""".strip()
        return QueryPlan(
            route="publication",
            retrieval_query=retrieval_query,
            route_reason="publication/external judgment question; answer must be cautious",
            external_caution=True,
        )

    if looks_like_innovation_query(user_query):
        retrieval_query = f"""
{user_query}
innovation novelty contribution main contribution related work positioning difference compared with prior work method conclusion limitation
创新点 创新性 主要贡献 相关工作 定位 相比已有方法 区别 方法 结论 局限
{profile_terms}
""".strip()
        return QueryPlan(
            route="innovation",
            retrieval_query=retrieval_query,
            route_reason="innovation/contribution question",
        )

    if is_broad_overview_query(user_query):
        retrieval_query = f"""
{user_query}
abstract introduction research motivation research objective research questions method framework experimental setup evaluation results discussion conclusion limitations future work main contribution
摘要 引言 研究问题 研究目标 方法框架 实验设置 主要结果 结论 局限性
{profile_terms}
""".strip()
        return QueryPlan(
            route="overview",
            retrieval_query=retrieval_query,
            route_reason="broad overview question",
        )

    if looks_like_claim_query(user_query):
        retrieval_query = f"""
{user_query}
evidence support not supported prove demonstrate compare limitation scope generalization result experiment conclusion
证据 支持 不支持 证明 对比 局限 范围 泛化 结果 实验 结论
{profile_terms}
""".strip()
        return QueryPlan(
            route="claim",
            retrieval_query=retrieval_query,
            route_reason="claim verification question",
        )

    if looks_like_experiment_query(user_query):
        retrieval_query = f"""
{user_query}
experiment setup result evaluation benchmark dataset metric table figure accuracy model comparison math500 GSM8K Qwen
实验 设置 结果 评测 基准 数据集 指标 表格 图 模型 对比
{profile_terms}
""".strip()
        return QueryPlan(
            route="experiment",
            retrieval_query=retrieval_query,
            route_reason="experiment/result question",
        )

    if looks_like_method_query(user_query):
        retrieval_query = f"""
{user_query}
method framework pipeline algorithm process key-step localization local repair Teacher Student Controller signal candidate selection
方法 框架 流程 算法 关键步骤定位 局部修复 教师 学生 控制器 信号 候选筛选
{profile_terms}
""".strip()
        return QueryPlan(
            route="method",
            retrieval_query=retrieval_query,
            route_reason="method/framework question",
        )

    profile_text = format_profile(corpus_profile)
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
5. If the query is in Chinese, keep important Chinese terms and add helpful English equivalents.
6. Return only the rewritten retrieval query, no explanation.

User query:
{user_query}

Rewritten retrieval query:
""".strip()

    response = str(llm.complete(prompt)).strip()
    rewritten = response if response else user_query
    return QueryPlan(
        route="general",
        retrieval_query=f"{user_query}\n{rewritten}",
        route_reason="general semantic question",
    )


def tokenize_for_sparse(text: str) -> List[str]:
    """Dependency-free bilingual tokenizer for BM25-style retrieval."""
    text = text.lower()
    tokens: List[str] = []
    tokens.extend(re.findall(r"[a-z0-9][a-z0-9_\-\.]*", text))
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    tokens.extend(chinese_chars)
    tokens.extend(chinese_chars[i] + chinese_chars[i + 1] for i in range(len(chinese_chars) - 1))
    return tokens


class SimpleBM25Retriever:
    def __init__(self, nodes: List[TextNode], k1: float = 1.5, b: float = 0.75):
        self.nodes = nodes
        self.k1 = k1
        self.b = b
        self.doc_tokens: List[List[str]] = []
        self.doc_len: List[int] = []
        self.df: Counter[str] = Counter()

        for node in nodes:
            metadata = node.metadata or {}
            metadata_text = " ".join(str(v) for v in metadata.values() if v is not None)
            text = f"{metadata_text}\n{node.get_content()}"
            tokens = tokenize_for_sparse(text)
            self.doc_tokens.append(tokens)
            self.doc_len.append(len(tokens))
            for token in set(tokens):
                self.df[token] += 1

        self.num_docs = len(nodes)
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)

    def score_document(self, query_tokens: List[str], doc_index: int) -> float:
        tokens = self.doc_tokens[doc_index]
        if not tokens:
            return 0.0

        tf = Counter(tokens)
        dl = self.doc_len[doc_index]
        score = 0.0

        for token in query_tokens:
            if token not in tf:
                continue
            df = self.df.get(token, 0)
            if df == 0:
                continue
            idf = math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))
            freq = tf[token]
            denom = freq + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
            score += idf * (freq * (self.k1 + 1)) / denom

        return score

    def retrieve(self, query: str, top_k: int) -> List[NodeWithScore]:
        query_tokens = tokenize_for_sparse(query)
        if not query_tokens:
            return []

        scored: List[Tuple[int, float]] = []
        for i in range(len(self.nodes)):
            score = self.score_document(query_tokens, i)
            if score > 0:
                scored.append((i, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [NodeWithScore(node=self.nodes[i], score=score) for i, score in scored[:top_k]]


def get_all_text_nodes(storage_context: StorageContext) -> List[TextNode]:
    docs = getattr(storage_context.docstore, "docs", {}) or {}
    return [node for node in docs.values() if isinstance(node, TextNode)]


def sorted_nodes_by_chunk(nodes: List[TextNode]) -> List[TextNode]:
    def key_func(node: TextNode):
        metadata = node.metadata or {}
        doc_id = str(metadata.get("doc_id") or "")
        try:
            page = int(metadata.get("page") or 999999)
        except Exception:
            page = 999999
        try:
            chunk_id = int(metadata.get("chunk_id") or 999999)
        except Exception:
            chunk_id = 999999
        return (doc_id, page, chunk_id)

    return sorted(nodes, key=key_func)


def node_to_scored(node: TextNode, score: float = 1.0) -> NodeWithScore:
    return NodeWithScore(node=node, score=score)


def node_key(item: NodeWithScore) -> str:
    node = item.node
    metadata = node.metadata or {}
    chunk_id = metadata.get("chunk_id")
    doc_id = metadata.get("doc_id")
    page = metadata.get("page")

    if doc_id is not None and chunk_id is not None:
        return f"{doc_id}:{chunk_id}"
    if page is not None and chunk_id is not None:
        return f"{page}:{chunk_id}"
    return node.node_id


def reciprocal_rank_fusion(
    dense_nodes: List[NodeWithScore],
    sparse_nodes: List[NodeWithScore],
    top_k: int,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.2,
) -> List[NodeWithScore]:
    fused_scores: Dict[str, float] = defaultdict(float)
    node_map: Dict[str, NodeWithScore] = {}

    for rank, item in enumerate(dense_nodes, start=1):
        key = node_key(item)
        node_map[key] = item
        fused_scores[key] += dense_weight / (rrf_k + rank)

    for rank, item in enumerate(sparse_nodes, start=1):
        key = node_key(item)
        node_map[key] = item
        fused_scores[key] += sparse_weight / (rrf_k + rank)

    ranked_keys = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)
    return [NodeWithScore(node=node_map[key].node, score=fused_scores[key]) for key in ranked_keys[:top_k]]


def node_section(node_with_score: NodeWithScore) -> str:
    metadata = node_with_score.node.metadata or {}
    return str(metadata.get("section_type") or "").strip().lower()


def node_page(node_with_score: NodeWithScore) -> int:
    metadata = node_with_score.node.metadata or {}
    try:
        return int(metadata.get("page") or 999999)
    except Exception:
        return 999999


def node_chunk_id(node_with_score: NodeWithScore) -> int:
    metadata = node_with_score.node.metadata or {}
    try:
        return int(metadata.get("chunk_id") or 999999)
    except Exception:
        return 999999


def node_char_len(node_with_score: NodeWithScore) -> int:
    metadata = node_with_score.node.metadata or {}
    try:
        return int(metadata.get("char_len") or len(clean_text(node_with_score.node.get_content())))
    except Exception:
        return len(clean_text(node_with_score.node.get_content()))


def collect_metadata_candidate_nodes(
    all_nodes: List[TextNode],
    sparse_retriever: SimpleBM25Retriever,
    retrieval_query: str,
    top_k: int,
    front_pages: int = 6,
) -> List[NodeWithScore]:
    """
    Generic front-matter candidate collector.

    It collects early pages plus sparse hits and their neighbors. The answer is
    produced by the LLM from these sources, not by fixed field regex extraction.
    """
    ordered_nodes = sorted_nodes_by_chunk(all_nodes)
    candidates: List[NodeWithScore] = []

    for node in ordered_nodes:
        metadata = node.metadata or {}
        try:
            page = int(metadata.get("page") or 999999)
        except Exception:
            page = 999999
        if page <= front_pages:
            candidates.append(node_to_scored(node, 1.0))

    sparse_hits = sparse_retriever.retrieve(retrieval_query, top_k=max(top_k * 4, 20))
    id_to_index = {node.node_id: i for i, node in enumerate(ordered_nodes)}

    for hit in sparse_hits:
        candidates.append(NodeWithScore(node=hit.node, score=float(hit.score or 0.0)))
        idx = id_to_index.get(hit.node.node_id)
        if idx is None:
            continue
        for nearby in ordered_nodes[max(0, idx - 2): min(len(ordered_nodes), idx + 3)]:
            candidates.append(node_to_scored(nearby, 0.9))

    seen = set()
    deduped: List[NodeWithScore] = []
    for item in sorted(candidates, key=lambda x: (node_page(x), node_chunk_id(x))):
        key = node_key(item)
        if key in seen:
            continue
        seen.add(key)
        if clean_text(item.node.get_content()):
            deduped.append(item)

    return deduped[: max(top_k, 14)]


def route_priority(route: str, section: str) -> int:
    return ROUTE_SECTION_PRIORITY.get(route, ROUTE_SECTION_PRIORITY["general"]).get(section, 9)


def filter_and_rank_nodes_by_route(
    nodes: List[NodeWithScore],
    route: str,
    top_k: int,
) -> List[NodeWithScore]:
    candidates: List[NodeWithScore] = []

    for item in nodes:
        section = node_section(item)
        metadata = item.node.metadata or {}
        unit_type = str(metadata.get("unit_type") or "").strip().lower()
        section_title = str(metadata.get("section_title") or "").strip().lower()
        text = clean_text(item.node.get_content())
        char_len = node_char_len(item)

        if route != "metadata" and section in NON_CONTENT_SECTIONS:
            continue

        if unit_type == "heading" and char_len < 120:
            continue

        if char_len < 80 and route != "metadata":
            continue

        noisy_title_patterns = [
            "thesis organization",
            "overall results .",
            "references",
            "acknowledgement",
            "致谢",
            "诚信承诺",
            "目录",
        ]
        if route != "metadata" and any(p in section_title for p in noisy_title_patterns):
            continue

        if text:
            candidates.append(item)

    if not candidates:
        candidates = nodes

    def rank_key(item: NodeWithScore):
        section = node_section(item)
        priority = route_priority(route, section)
        score = float(item.score or 0.0)
        return (priority, -score, node_page(item), node_chunk_id(item))

    return sorted(candidates, key=rank_key)[:top_k]


def retrieve_nodes(
    dense_retriever: Any,
    sparse_retriever: SimpleBM25Retriever,
    retrieval_query: str,
    query_plan: QueryPlan,
    top_k: int,
    retrieval_mode: str = "hybrid",
) -> List[NodeWithScore]:
    retrieval_mode = retrieval_mode.lower().strip()
    candidate_k = max(top_k * 6, 40)

    dense_nodes: List[NodeWithScore] = []
    sparse_nodes: List[NodeWithScore] = []

    if retrieval_mode in {"dense", "hybrid"}:
        dense_nodes = dense_retriever.retrieve(retrieval_query)

    if retrieval_mode in {"sparse", "hybrid"}:
        sparse_nodes = sparse_retriever.retrieve(retrieval_query, top_k=candidate_k)

    if retrieval_mode == "dense":
        nodes = dense_nodes[:candidate_k]
    elif retrieval_mode == "sparse":
        nodes = sparse_nodes[:candidate_k]
    elif retrieval_mode == "hybrid":
        nodes = reciprocal_rank_fusion(
            dense_nodes=dense_nodes,
            sparse_nodes=sparse_nodes,
            top_k=candidate_k,
            dense_weight=1.0,
            sparse_weight=1.2,
        )
    else:
        raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode}")

    return filter_and_rank_nodes_by_route(nodes, query_plan.route, top_k)


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


def answer_metadata_question_with_llm(
    llm: OpenAI,
    question: str,
    query_plan: QueryPlan,
    nodes: List[NodeWithScore],
) -> str:
    source_text = "\n\n".join(build_source_blocks(nodes))
    requested_attribute = query_plan.requested_attribute or "document metadata"

    prompt = f"""
You are a source-grounded metadata QA assistant.

The user asks about a metadata/front-matter attribute of a document.

Requested attribute:
{requested_attribute}

Use only the retrieved sources.
The sources may contain many nearby fields, such as title, author, advisor, department, program, date, university, signature, or affiliation.

Important rules:
1. Answer only the requested attribute.
2. Do not list nearby but different fields.
3. If the user asks for the author, do not include advisor, department, program, title, or date.
4. If the user asks for the advisor/supervisor, do not include author, department, program, title, or date.
5. If the user asks for program/major, do not answer the research topic unless the source gives the program/major.
6. If multiple plausible values appear, choose the value directly attached to the requested attribute and mention uncertainty only if needed.
7. If the sources do not clearly provide the requested attribute, say that the retrieved sources do not clearly provide it.
8. Do not use outside knowledge.

If the question is in Chinese, answer in Chinese.
If the question is in English, answer in English.

User question:
{question}

Retrieved sources:
{source_text}

Answer:
""".strip()

    return str(llm.complete(prompt)).strip()


def generate_answer(llm: OpenAI, question: str, query_plan: QueryPlan, nodes: List[NodeWithScore]) -> str:
    source_text = "\n\n".join(build_source_blocks(nodes))
    route = query_plan.route

    if route == "overview":
        task_instruction = """
The user asks for a broad overview of the document.
Focus on: research problem, motivation, method/framework, experimental setup, main results, conclusion, and limitations.
Do not treat acknowledgements, cover page, commitment statement, table of contents, or references as the main content.
""".strip()
    elif route == "innovation":
        task_instruction = """
The user asks about innovation, novelty, contribution, or relationship to prior work.
Focus on abstract, introduction, related work positioning, method, and conclusion.
Answer what is directly supported by the sources.
If the document does not state that it is based on one specific prior paper, do not invent one. Instead, summarize the related research directions or prior-work categories mentioned in the sources.
""".strip()
    elif route == "publication":
        task_instruction = """
The user asks partly about innovation/contribution and partly about publication or venue suitability.
Use the PDF sources to discuss the paper's topic, contribution, evidence strength, and limitations.
Do not claim whether it can be accepted by a specific conference unless the sources explicitly provide that information.
Say that venue suitability requires external information such as conference scope, recent accepted papers, novelty standard, and experimental strength.
""".strip()
    elif route == "method":
        task_instruction = """
The user asks about method, framework, mechanism, or implementation.
Explain the method using only the retrieved sources, and cite the relevant sources.
""".strip()
    elif route == "experiment":
        task_instruction = """
The user asks about experiment, results, metrics, tables, figures, datasets, or models.
Prioritize exact numbers, table/figure names, settings, and limitations when present.
If exact values are not in the retrieved sources, say so.
""".strip()
    elif route == "claim":
        task_instruction = """
The user asks whether a claim is supported, proved, compared, reported, or generalized.
Answer yes only if the retrieved sources explicitly support it.
If the evidence is partial or absent, clearly say what is supported and what is not supported.
""".strip()
    else:
        task_instruction = """
Answer the user question using only the retrieved sources.
Be concise, source-grounded, and cautious about unsupported claims.
""".strip()

    prompt = f"""
You are a source-grounded QA assistant for a RAG demo.

Query route:
{route}

Task instruction:
{task_instruction}

General grounding rules:
1. Do not use outside knowledge.
2. Cite the retrieved sources in natural language, for example "Source 2" or "Sources 1 and 4".
3. If the sources do not support a claim, say clearly that the retrieved sources do not support it.
4. Do not turn absence of evidence into a positive claim.
5. If the question is in Chinese, answer in Chinese. If the question is in English, answer in English.

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


def print_demo_summary(question: str, query_plan: QueryPlan, show_rewritten_query: bool, retrieval_mode: str) -> None:
    print("\nQuestion")
    print("--------")
    print(question)

    if show_rewritten_query:
        print("\nRetrieval Query")
        print("---------------")
        print(f"[route={query_plan.route}; mode={retrieval_mode}; reason={query_plan.route_reason}]")
        print(query_plan.retrieval_query)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*", help="Question to ask the RAG system.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--corpus_profile", default=None)
    parser.add_argument("--show_rewritten_query", action="store_true")
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument(
        "--retrieval_mode",
        choices=["dense", "sparse", "hybrid"],
        default="hybrid",
        help="Retrieval mode. Default is hybrid.",
    )
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

    candidate_k = max(top_k * 6, 40)
    dense_retriever = index.as_retriever(similarity_top_k=candidate_k)

    all_nodes = get_all_text_nodes(storage_context)
    sparse_retriever = SimpleBM25Retriever(all_nodes)

    query_plan = classify_query(llm=llm, user_query=question, corpus_profile=corpus_profile)

    if query_plan.route == "metadata":
        nodes = collect_metadata_candidate_nodes(
            all_nodes=all_nodes,
            sparse_retriever=sparse_retriever,
            retrieval_query=query_plan.retrieval_query,
            top_k=top_k,
            front_pages=6,
        )
        answer = answer_metadata_question_with_llm(
            llm=llm,
            question=question,
            query_plan=query_plan,
            nodes=nodes,
        )
        output_mode = f"{args.retrieval_mode} + generic_front_matter_metadata_qa"
    else:
        nodes = retrieve_nodes(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            retrieval_query=query_plan.retrieval_query,
            query_plan=query_plan,
            top_k=top_k,
            retrieval_mode=args.retrieval_mode,
        )
        answer = generate_answer(llm=llm, question=question, query_plan=query_plan, nodes=nodes)
        output_mode = f"{args.retrieval_mode} + route_aware_retrieval"

    print_demo_summary(
        question=question,
        query_plan=query_plan,
        show_rewritten_query=args.show_rewritten_query,
        retrieval_mode=output_mode,
    )
    print_answer(answer)
    print_evidence(nodes)


if __name__ == "__main__":
    main()
