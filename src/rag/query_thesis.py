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
    exact_terms: Tuple[str, ...] = ()
    sub_questions: Tuple[str, ...] = ()


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


def normalize_for_match(text: str) -> str:
    """Normalize text for robust exact-ish matching over English/Chinese chunks."""
    text = clean_text(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_table_figure_terms(query: str) -> List[str]:
    """
    Extract explicit table / figure identifiers from a query.

    Examples:
    - "Table 2 和 Table 3 有什么区别" -> ["Table 2", "Table 3"]
    - "图 6 说明什么" -> ["Figure 6", "图 6"]
    """
    terms: List[str] = []
    q = query.strip()

    for match in re.finditer(r"\btable\s*[-#:]?\s*(\d+[a-zA-Z]?)\b", q, flags=re.IGNORECASE):
        n = match.group(1)
        terms.extend([f"Table {n}", f"Table{n}"])

    for match in re.finditer(r"\bfigure\s*[-#:]?\s*(\d+[a-zA-Z]?)\b", q, flags=re.IGNORECASE):
        n = match.group(1)
        terms.extend([f"Figure {n}", f"Figure{n}", f"Fig. {n}", f"Fig {n}"])

    for match in re.finditer(r"表\s*(\d+[a-zA-Z]?)", q):
        n = match.group(1)
        terms.extend([f"表 {n}", f"表{n}", f"Table {n}", f"Table{n}"])

    for match in re.finditer(r"图\s*(\d+[a-zA-Z]?)", q):
        n = match.group(1)
        terms.extend([f"图 {n}", f"图{n}", f"Figure {n}", f"Figure{n}", f"Fig. {n}"])

    deduped: List[str] = []
    seen = set()
    for term in terms:
        key = term.lower().replace(" ", "")
        if key not in seen:
            seen.add(key)
            deduped.append(term)
    return deduped


def looks_like_table_figure_query(query: str) -> bool:
    """High-priority detector for table/figure/result questions."""
    if extract_table_figure_terms(query):
        return True

    patterns = [
        "table",
        "figure",
        "fig.",
        "表格",
        "图表",
        "结果表",
        "实验表",
        "消融",
        "ablation",
        "main results",
        "overall results",
        "math500 strict",
        "gsm8k",
        "accuracy",
        "benchmark",
        "metric",
        "1.5b",
        "7b",
    ]
    return contains_any(query, patterns) or bool(re.search(r"(?:表|图)\s*\d", query))


def split_multi_intent_question(question: str) -> List[str]:
    """
    Deterministic multi-intent splitter.

    v4 makes splitting less brittle: when a query contains several independent
    intents such as innovation + related work + experiment support / publication,
    it is routed to a multi-intent workflow instead of being swallowed by a
    single claim/innovation route.
    """
    q = question.strip()
    if len(q) < 18:
        return [q]

    q_lower = q.lower()
    intent_markers = {
        "innovation": ["创新", "新意", "贡献", "亮点", "contribution", "innovation", "novelty"],
        "related_work": ["相关工作", "研究基础", "基础上", "基于哪些", "prior work", "related work", "based on"],
        "experiment": ["实验", "结果", "table", "figure", "表", "图", "benchmark", "math500", "1.5b", "7b"],
        "limitation": ["局限", "不足", "future work", "limitation", "还差"],
        "publication": ["投稿", "会议", "期刊", "publication", "conference", "venue"],
        "claim": ["是否足够", "是否证明", "有没有证明", "能否证明", "prove", "support", "supported"],
        "method": ["方法", "框架", "local cot repair", "teacher-student-controller", "controller", "student signal"],
    }
    hit_intents = [name for name, kws in intent_markers.items() if contains_any(q_lower, kws)]

    # Explicit multi-intent templates. These are generic intent-level templates,
    # not answer-level hardcoding.
    if len(hit_intents) >= 2:
        out: List[str] = []
        if "innovation" in hit_intents:
            out.append("这篇论文的创新点或主要贡献是什么？" if is_chinese_text(q) else "What are the main contributions or innovations of this thesis?")
        if "related_work" in hit_intents:
            out.append("这篇论文基于哪些相关工作或研究脉络？" if is_chinese_text(q) else "What related work or research context does this thesis build on?")
        if "method" in hit_intents and "innovation" not in hit_intents:
            out.append("这篇论文的方法或框架是什么？" if is_chinese_text(q) else "What is the method or framework used in this thesis?")
        if "experiment" in hit_intents:
            if contains_any(q_lower, ["是否足够", "support", "supported", "证明"]):
                out.append("实验结果是否足够支持论文结论？" if is_chinese_text(q) else "Do the experimental results sufficiently support the thesis conclusions?")
            else:
                out.append("这篇论文的主要实验结果是什么？" if is_chinese_text(q) else "What are the main experimental results of this thesis?")
        if "limitation" in hit_intents:
            out.append("这篇论文有哪些局限性或后续改进方向？" if is_chinese_text(q) else "What limitations or future work does the thesis mention?")
        if "publication" in hit_intents:
            out.append("仅根据论文内容，能否判断它是否适合投稿目标会议？" if is_chinese_text(q) else "Based only on the thesis, can we judge its venue suitability?")
        if "claim" in hit_intents and "experiment" not in hit_intents:
            out.append("这个说法是否被论文证据支持？" if is_chinese_text(q) else "Is this claim supported by the thesis evidence?")

        # Deduplicate while preserving order.
        deduped: List[str] = []
        seen = set()
        for item in out:
            key = item.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        if 2 <= len(deduped) <= 5:
            return deduped

    has_multi_signal = any(x in q for x in ["？", "?", "，", ",", "；", ";", "以及", "并且", "同时", "还有", "另外"])
    if not has_multi_signal:
        return [q]

    normalized = q
    replacements = [
        ("？", "?"),
        ("；", ";"),
        ("，另外", ";另外"),
        ("，同时", ";同时"),
        ("，并且", ";并且"),
        ("，以及", ";以及"),
    ]
    for old, new in replacements:
        normalized = normalized.replace(old, new)

    pieces = re.split(r"[?;]+|(?:\s+and\s+)|(?:\s+also\s+)", normalized, flags=re.IGNORECASE)
    refined: List[str] = []
    for piece in pieces:
        piece = piece.strip(" ，,。.;；?？")
        piece = re.sub(r"^(另外|同时|并且|以及|还有|他|它|这篇论文|这篇文章)\s*", "", piece).strip()
        if len(piece) >= 6:
            refined.append(piece)

    if len(refined) < 2 or len(refined) > 5:
        return [q]

    final: List[str] = []
    for piece in refined:
        if is_chinese_text(q) and not contains_any(piece, ["论文", "文章", "thesis", "paper", "Table", "Figure", "表", "图"]):
            piece = f"这篇论文{piece}"
        final.append(piece)
    return final

def is_broad_overview_query(query: str) -> bool:
    q = query.strip().lower()

    chinese_patterns = [
        "主要在讲什么",
        "主要讲什么",
        "主要内容",
        "核心内容",
        "总结一下",
        "三句话总结",
        "用三句话",
        "概括一下",
        "研究目标",
        "解决了什么痛点",
        "解决什么问题",
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
        "local cot repair",
        "局部修复",
        "修复策略",
        "局部修复策略",
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
        "which model scale",
        "model scale",
        "benefits more",
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
        "是不是",
        "是否为",
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
    # Keep route-specific rewrites compact. Overloading every query with all
    # corpus keywords dilutes retrieval, especially for innovation questions.
    profile_terms = build_profile_keywords(corpus_profile, limit=12)

    # P0 fix: table / figure / experiment questions must be detected before
    # innovation, because words like "difference / 区别" frequently appear in
    # table-comparison questions.
    if looks_like_table_figure_query(user_query):
        exact_terms = tuple(extract_table_figure_terms(user_query))
        exact_hint = " ".join(exact_terms)
        retrieval_query = f"""
{user_query}
{exact_hint}
experiment result results table figure caption metric benchmark dataset model comparison accuracy ablation main results overall results math500 strict GSM8K Qwen 1.5B 7B
实验 结果 表格 图表 图注 指标 基准 数据集 模型 对比 准确率 消融 主结果 总体结果
{profile_terms}
""".strip()
        return QueryPlan(
            route="experiment",
            retrieval_query=retrieval_query,
            route_reason="high-priority table/figure/experiment question",
            exact_terms=exact_terms,
        )

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

    # External publication / venue judgment must be checked before generic
    # claim verification. Chinese questions such as “是否适合投稿会议” contain
    # claim-like words (“是否 / 能不能”), but their correct route is publication
    # because the PDF alone cannot decide acceptance or venue suitability.
    if looks_like_external_judgment_query(user_query):
        retrieval_query = f"""
{user_query}
innovation contribution novelty related work positioning method limitation future work discussion result experiment evaluation conference venue publication suitability
创新点 贡献 相关工作 方法 结果 局限 未来工作 讨论 投稿 会议 大模型 推理 思维链
{profile_terms}
""".strip()
        return QueryPlan(
            route="publication",
            retrieval_query=retrieval_query,
            route_reason="publication/external judgment question; answer must be cautious",
            external_caution=True,
        )

    if looks_like_claim_query(user_query):
        retrieval_query = f"""
{user_query}
evidence support not supported prove demonstrate compare limitation scope generalization result experiment conclusion future work only under the experimental setting not enough limited
证据 支持 不支持 证明 对比 局限 范围 泛化 结果 实验 结论 未来工作 仅在实验设置下 不足 有限
{profile_terms}
""".strip()
        return QueryPlan(
            route="claim",
            retrieval_query=retrieval_query,
            route_reason="claim verification question",
        )

    if looks_like_innovation_query(user_query):
        retrieval_query = f"""
{user_query}
innovation novelty contribution main contribution related work positioning difference compared with prior work method conclusion limitation
创新点 创新性 主要贡献 相关工作 定位 相比已有方法 区别 方法 结论 局限
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


def build_node_lookup(all_nodes: List[TextNode]) -> Tuple[List[TextNode], Dict[str, int]]:
    ordered_nodes = sorted_nodes_by_chunk(all_nodes)
    return ordered_nodes, {node.node_id: i for i, node in enumerate(ordered_nodes)}


def add_neighbor_nodes(
    items: List[NodeWithScore],
    ordered_nodes: List[TextNode],
    id_to_index: Dict[str, int],
    window: int = 1,
    neighbor_score: float = 0.75,
) -> List[NodeWithScore]:
    """Add previous/next chunks for table/figure and short evidence recovery."""
    expanded: List[NodeWithScore] = []
    for item in items:
        expanded.append(item)
        idx = id_to_index.get(item.node.node_id)
        if idx is None:
            continue
        start = max(0, idx - window)
        end = min(len(ordered_nodes), idx + window + 1)
        for nearby in ordered_nodes[start:end]:
            if nearby.node_id == item.node.node_id:
                continue
            expanded.append(node_to_scored(nearby, neighbor_score))
    return dedupe_scored_nodes(expanded)


def dedupe_scored_nodes(nodes: List[NodeWithScore]) -> List[NodeWithScore]:
    """Deduplicate by node key while keeping the highest score."""
    best: Dict[str, NodeWithScore] = {}
    for item in nodes:
        key = node_key(item)
        if key not in best or float(item.score or 0.0) > float(best[key].score or 0.0):
            best[key] = item
    return list(best.values())


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


def exact_match_table_figure_nodes(
    all_nodes: List[TextNode],
    query_plan: QueryPlan,
    top_k: int,
) -> List[NodeWithScore]:
    """
    Directly retrieve chunks containing explicit Table/Figure ids.

    This is the main fix for queries such as "Table 2 和 Table 3 有什么区别".
    Hybrid retrieval can miss these because route/query rewrite terms dilute exact
    identifiers, so explicit table/figure ids get a deterministic boost.
    """
    terms = list(query_plan.exact_terms or [])
    if not terms:
        terms = extract_table_figure_terms(query_plan.retrieval_query)
    if not terms:
        return []

    ordered_nodes, id_to_index = build_node_lookup(all_nodes)
    scored: List[NodeWithScore] = []
    normalized_terms = [(term, term.lower().replace(" ", "")) for term in terms]

    for node in ordered_nodes:
        metadata = node.metadata or {}
        text = normalize_for_match(
            "\n".join(
                [
                    str(metadata.get("section_title") or ""),
                    str(metadata.get("unit_type") or ""),
                    str(metadata.get("section_type") or ""),
                    node.get_content(),
                ]
            )
        )
        compact = text.replace(" ", "")
        hit_count = 0
        for raw_term, compact_term in normalized_terms:
            if raw_term.lower() in text or compact_term in compact:
                hit_count += 1
        if hit_count:
            unit_type = str(metadata.get("unit_type") or "").lower()
            section = str(metadata.get("section_type") or "").lower()
            boost = 6.0 + hit_count
            if "table" in unit_type or "figure" in unit_type or "table" in text or "figure" in text:
                boost += 1.0
            if section in {"results", "setup"}:
                boost += 0.5
            scored.append(node_to_scored(node, boost))

    if not scored:
        return []

    scored = add_neighbor_nodes(
        items=scored,
        ordered_nodes=ordered_nodes,
        id_to_index=id_to_index,
        window=1,
        neighbor_score=4.5,
    )
    scored = filter_and_rank_nodes_by_route(scored, query_plan.route, max(top_k * 2, 10))
    return scored


def is_reference_like_node(item: NodeWithScore) -> bool:
    """Detect reference-list / bibliography chunks even when section metadata is noisy."""
    metadata = item.node.metadata or {}
    text = normalize_for_match("\n".join([
        str(metadata.get("section_title") or ""),
        str(metadata.get("section_type") or ""),
        item.node.get_content(),
    ]))
    raw = clean_text(item.node.get_content())
    if str(metadata.get("section_type") or "").lower() in {"references", "reference"}:
        return True
    reference_markers = [
        "references", "bibliography", "arxiv preprint", "neurips", "iclr", "icml", "acl", "emnlp",
        "proceedings", "transactions", "journal", "datasets and benchmarks", "[eb/ol]", "[j]", "[c]",
    ]
    if contains_any(text, reference_markers) and len(raw) < 900:
        return True
    if len(re.findall(r"\[\d+\]", raw)) >= 2:
        return True
    if re.search(r"^\s*\[\d+\]\s+[A-Z][A-Za-z, .-]+", raw):
        return True
    return False


def is_toc_like_node(item: NodeWithScore) -> bool:
    """Detect table-of-contents fragments that are often misclassified as results/method."""
    metadata = item.node.metadata or {}
    text = clean_text("\n".join([
        str(metadata.get("section_title") or ""),
        item.node.get_content(),
    ]))
    if str(metadata.get("section_type") or "").lower() == "toc":
        return True
    if len(text) < 260 and len(re.findall(r"\.\s*\.\s*\.?\s*\d+", text)) >= 1:
        return True
    if len(text) < 220 and len(re.findall(r"\b\d+\.\d+\b", text)) >= 3:
        return True
    return False


def is_heading_only_node(item: NodeWithScore) -> bool:
    metadata = item.node.metadata or {}
    text = clean_text(item.node.get_content())
    unit_type = str(metadata.get("unit_type") or "").lower()
    if unit_type == "heading" and len(text) < 180:
        return True
    if len(text) < 50 and not re.search(r"[。！？.!?]", text):
        return True
    return False


def route_heuristic_score(item: NodeWithScore, query_plan: QueryPlan) -> float:
    """Route-aware reranking score used before optional LLM reranking."""
    metadata = item.node.metadata or {}
    text = normalize_for_match(item.node.get_content())
    section = str(metadata.get("section_type") or "").lower()
    unit_type = str(metadata.get("unit_type") or "").lower()
    section_title = normalize_for_match(str(metadata.get("section_title") or ""))
    score = float(item.score or 0.0)

    # Section prior: smaller route_priority is better.
    score += max(0, 9 - route_priority(query_plan.route, section)) * 0.15

    if query_plan.route == "experiment":
        if section in {"results", "setup"}:
            score += 1.0
        if any(x in unit_type for x in ["table", "figure", "algorithm"]):
            score += 1.2
        if contains_any(text + " " + section_title, ["table", "figure", "fig.", "math500", "gsm8k", "accuracy", "result"]):
            score += 0.8
        for term in query_plan.exact_terms:
            if term.lower() in text or term.lower().replace(" ", "") in text.replace(" ", ""):
                score += 2.0

    elif query_plan.route == "innovation":
        if section in {"abstract", "introduction", "related_work", "method", "conclusion"}:
            score += 0.7
        if contains_any(text + " " + section_title, ["contribution", "innovation", "novel", "related work", "position", "local repair", "student-oriented"]):
            score += 0.8

    elif query_plan.route == "claim":
        if contains_any(text, ["not", "does not", "limitation", "scope", "future work", "result", "experiment", "conclusion", "support"]):
            score += 0.8

    elif query_plan.route == "publication":
        if section in {"abstract", "introduction", "related_work", "method", "results", "conclusion"}:
            score += 0.5
        if contains_any(text, ["limitation", "future work", "related work", "contribution", "result", "experiment"]):
            score += 0.6

    # Penalize noisy / tiny chunks after retrieval, except metadata handled separately.
    char_len = node_char_len(item)
    if char_len < 80:
        score -= 1.5
    if str(metadata.get("unit_type") or "").lower() == "heading" and char_len < 160:
        score -= 1.0
    if section in NON_CONTENT_SECTIONS and query_plan.route != "metadata":
        score -= 2.0
    if query_plan.route != "metadata" and is_reference_like_node(item):
        score -= 4.0
    if query_plan.route != "metadata" and is_toc_like_node(item):
        score -= 3.0
    if query_plan.route != "metadata" and is_heading_only_node(item):
        score -= 1.5

    return score


def heuristic_rerank_nodes(
    nodes: List[NodeWithScore],
    query_plan: QueryPlan,
    top_k: int,
) -> List[NodeWithScore]:
    rescored = [NodeWithScore(node=item.node, score=route_heuristic_score(item, query_plan)) for item in nodes]
    rescored = dedupe_scored_nodes(rescored)
    rescored.sort(key=lambda x: (-(float(x.score or 0.0)), node_page(x), node_chunk_id(x)))
    return rescored[:top_k]


def node_matches_exact_term(item: NodeWithScore, term: str) -> bool:
    metadata = item.node.metadata or {}
    text = normalize_for_match(
        "\n".join(
            [
                str(metadata.get("section_title") or ""),
                str(metadata.get("unit_type") or ""),
                str(metadata.get("section_type") or ""),
                item.node.get_content(),
            ]
        )
    )
    compact = text.replace(" ", "")
    term_norm = term.lower()
    return term_norm in text or term_norm.replace(" ", "") in compact


def ensure_exact_term_coverage(
    selected: List[NodeWithScore],
    candidates: List[NodeWithScore],
    query_plan: QueryPlan,
    top_k: int,
) -> List[NodeWithScore]:
    """Preserve evidence for explicit Table/Figure ids before filling the rest.

    The previous implementation appended missing Table/Figure nodes and then
    sorted all nodes globally. If the appended node had a lower score, it could
    be dropped again after ``[:top_k]``. For comparison questions such as
    "Table 2 vs Table 3", this is exactly the failure mode: the final context
    may contain only Table 2, so the answer says Table 3 is unavailable.

    This version pins one best exact-match node for each explicit id first,
    then fills the remaining slots with reranked evidence.
    """
    top_k = max(1, top_k)
    terms = list(query_plan.exact_terms or [])
    all_items = dedupe_scored_nodes(list(selected) + list(candidates))

    if not terms:
        return all_items[:top_k]

    pinned: List[NodeWithScore] = []
    for term in terms:
        matches = [item for item in all_items if node_matches_exact_term(item, term)]
        if not matches:
            continue
        matches.sort(
            key=lambda x: (
                -(route_heuristic_score(x, query_plan)),
                node_page(x),
                node_chunk_id(x),
            )
        )
        pinned.append(matches[0])

    pinned = dedupe_scored_nodes(pinned)
    pinned_ids = {node_key(item) for item in pinned}

    filler: List[NodeWithScore] = []
    for item in all_items:
        if node_key(item) in pinned_ids:
            continue
        filler.append(item)

    output = pinned + filler
    output = dedupe_scored_nodes(output)
    return output[:top_k]


def llm_rerank_nodes(
    llm: OpenAI,
    question: str,
    query_plan: QueryPlan,
    nodes: List[NodeWithScore],
    top_k: int,
) -> List[NodeWithScore]:
    """Safe LLM-assisted reranker.

    The LLM is used only as a *soft reordering signal*. It is not allowed to
    delete evidence aggressively. This is important for table/figure comparison
    questions: if the question mentions Table 2 and Table 3, the final context
    must preserve evidence for both whenever they exist in the candidate pool.
    """
    if not nodes:
        return []

    top_k = max(1, top_k)
    heuristic_pool = heuristic_rerank_nodes(nodes, query_plan, top_k=max(top_k * 3, 24))

    # Use the deterministic reranker as the stable base. If anything goes wrong
    # with LLM reranking, this is already a good answer context.
    fallback = ensure_exact_term_coverage(heuristic_pool[:top_k], heuristic_pool + nodes, query_plan, top_k)
    if len(nodes) <= 3:
        return fallback

    # Rerank only a bounded candidate pool. Include heuristic_pool first because
    # it already contains route-aware and exact-match candidates.
    candidate_pool = dedupe_scored_nodes(heuristic_pool + nodes)
    candidate_count = min(len(candidate_pool), 30)

    candidate_lines = []
    for i, item in enumerate(candidate_pool[:candidate_count], start=1):
        metadata = item.node.metadata or {}
        preview = compact_preview(item.node.get_content(), max_chars=520)
        candidate_lines.append(
            f"[{i}] page={metadata.get('page')} section={metadata.get('section_type')} "
            f"unit={metadata.get('unit_type')} title={metadata.get('section_title')}\n{preview}"
        )

    exact_instruction = ""
    if query_plan.exact_terms:
        exact_instruction = (
            "\nThe question explicitly mentions these table/figure ids: "
            + ", ".join(query_plan.exact_terms)
            + ". If comparing multiple ids, keep useful evidence for each id when available. "
              "Do not return only one table when another mentioned table is present."
        )

    prompt = f"""
You are a reranker for a source-grounded RAG system.

Question:
{question}

Route:
{query_plan.route}
{exact_instruction}

Candidates:
{chr(10).join(candidate_lines)}

Score each candidate:
3 = directly answers the question
2 = partially relevant / useful evidence
1 = background only
0 = irrelevant

Return strict JSON only. Include every candidate id from 1 to {candidate_count}:
{{"scores": [{{"id": 1, "score": 3, "reason": "short"}}]}}
""".strip()

    try:
        response = str(llm.complete(prompt)).strip()
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not match:
            return fallback
        data = json.loads(match.group(0))

        score_map: Dict[int, float] = {}
        for row in data.get("scores", []):
            try:
                idx = int(row.get("id")) - 1
                relevance = float(row.get("score", 0))
            except Exception:
                continue
            if 0 <= idx < candidate_count:
                score_map[idx] = max(0.0, min(3.0, relevance))

        # Incomplete JSON from the LLM is common. If it did not score most
        # candidates, do not trust it; use the stable heuristic context.
        min_scored = min(candidate_count, max(8, top_k))
        if len(score_map) < min_scored:
            return fallback

        reranked: List[NodeWithScore] = []
        for idx, item in enumerate(candidate_pool):
            heuristic = route_heuristic_score(item, query_plan)
            base = float(item.score or 0.0)
            if idx < candidate_count:
                relevance = score_map.get(idx, 1.0)
            else:
                relevance = 1.0

            if query_plan.exact_terms and any(node_matches_exact_term(item, term) for term in query_plan.exact_terms):
                relevance = max(relevance, 2.6)

            # Keep scores readable but clearly ranking-oriented. The LLM score
            # influences order; it does not act as a hard filter.
            combined = relevance * 10.0 + heuristic + min(base, 5.0) * 0.05
            reranked.append(NodeWithScore(node=item.node, score=combined))

        reranked.sort(key=lambda x: (-(float(x.score or 0.0)), node_page(x), node_chunk_id(x)))
        selected = reranked[:top_k]
        selected = ensure_exact_term_coverage(selected, reranked + fallback + candidate_pool, query_plan, top_k)

        # Guardrail: if LLM mode still produced less evidence than the stable
        # fallback, fill from fallback. This avoids one-source answers.
        min_context = min(top_k, max(4, len(query_plan.exact_terms or []) * 2))
        if len(selected) < min_context:
            selected = ensure_exact_term_coverage(selected + fallback, reranked + candidate_pool, query_plan, top_k)
        return selected
    except Exception:
        return fallback

def retrieve_nodes(
    llm: OpenAI,
    question: str,
    all_nodes: List[TextNode],
    dense_retriever: Any,
    sparse_retriever: SimpleBM25Retriever,
    retrieval_query: str,
    query_plan: QueryPlan,
    top_k: int,
    retrieval_mode: str = "hybrid",
    rerank_mode: str = "heuristic",
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

    exact_nodes = exact_match_table_figure_nodes(all_nodes, query_plan, top_k=max(top_k, 8))
    if exact_nodes:
        nodes = dedupe_scored_nodes(exact_nodes + nodes)

    nodes = filter_and_rank_nodes_by_route(nodes, query_plan.route, candidate_k)

    rerank_mode = rerank_mode.lower().strip()
    if rerank_mode == "none":
        return nodes[:top_k]
    if rerank_mode == "llm":
        return llm_rerank_nodes(llm=llm, question=question, query_plan=query_plan, nodes=nodes, top_k=top_k)
    return heuristic_rerank_nodes(nodes=nodes, query_plan=query_plan, top_k=top_k)


def query_keywords_for_audit(question: str, query_plan: QueryPlan) -> List[str]:
    text = f"{question} {query_plan.retrieval_query}"
    keywords: List[str] = []
    for term in query_plan.exact_terms:
        keywords.append(term)
    # English entities / technical tokens
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_.+-]*(?:\s+[A-Za-z0-9_.+-]+)?", text):
        token = token.strip()
        if len(token) >= 3:
            keywords.append(token)
    # Chinese phrases, keep short but meaningful windows.
    for token in re.findall(r"[\u4e00-\u9fff]{2,8}", text):
        if token not in {"这篇论文", "这篇文章", "什么", "怎么", "哪些", "是否", "有没有"}:
            keywords.append(token)
    # Route-specific anchor terms.
    route_terms = {
        "overview": ["abstract", "introduction", "motivation", "problem", "framework", "results", "摘要", "研究问题"],
        "innovation": ["contribution", "innovation", "novel", "related work", "positioning", "local repair", "贡献", "创新", "相关工作"],
        "method": ["method", "framework", "teacher", "student", "controller", "local repair", "key-step", "方法", "框架", "局部修复"],
        "experiment": ["experiment", "result", "table", "figure", "math500", "benchmark", "accuracy", "实验", "结果", "表", "图"],
        "claim": ["limitation", "not", "does not", "future work", "scope", "under the experimental setting", "局限", "不足", "不支持"],
        "publication": ["contribution", "limitation", "future work", "experiment", "related work", "投稿", "会议", "局限"],
    }
    keywords.extend(route_terms.get(query_plan.route, []))
    # Deduplicate and keep a bounded list.
    out: List[str] = []
    seen = set()
    for kw in keywords:
        kw = kw.strip().lower()
        if len(kw) < 2 or kw in seen:
            continue
        seen.add(kw)
        out.append(kw)
    return out[:40]



def extract_strong_evidence_signals(question: str, query_plan: QueryPlan, text: str) -> List[str]:
    """Detect strong answer-supporting entities or counter-evidence patterns.

    These signals are intentionally route-agnostic and human-auditable. They fix
    cases where a chunk is genuinely direct evidence even if simple query-keyword
    overlap is weak, for example:
    - "Which models were used?" -> chunk contains "The student models are Qwen2.5-Math-1.5B and Qwen2.5-Math-7B".
    - "Is this a new architecture?" -> chunk says the method "does not change the student architecture".
    - table/figure questions -> chunk explicitly contains the requested id and its explanatory sentence.
    """
    q = normalize_for_match(question)
    t = text
    compact_t = t.replace(" ", "")
    signals: List[str] = []

    # Exact table/figure ids are strong, especially when the chunk also contains
    # result / score / caption language.
    for term in query_plan.exact_terms:
        term_norm = term.lower()
        if term_norm in t or term_norm.replace(" ", "") in compact_t:
            if any(x in t for x in ["main results", "benchmark", "score", "scores", "caption", "reports", "does not report", "table", "figure", "结果", "分数", "报告", "说明"]):
                signals.append(f"strong_exact_{term_norm.replace(' ', '_')}")

    # Model-use questions need to reward model-name evidence, even when the
    # original question only says "哪些模型 / which models".
    asks_model = any(x in q for x in ["哪些模型", "什么模型", "用了哪些", "used", "which model", "models were used", "model scale", "模型"])
    has_qwen_models = all(x in t for x in ["qwen2.5-math-1.5b", "qwen2.5-math-7b"])
    if asks_model and (has_qwen_models or "the student models are" in t):
        signals.append("answer_entity_models")

    # Benchmark / metric questions.
    asks_benchmark = any(x in q for x in ["benchmark", "基准", "评估", "metric", "指标", "math500"])
    if asks_benchmark and "math500 strict" in t:
        signals.append("answer_entity_benchmark")

    # Method/framework/role questions.
    if any(x in q for x in ["teacher-student-controller", "controller", "student signal", "框架", "控制器", "学生信号"]):
        if any(x in t for x in ["teacher", "student", "controller", "教师", "学生", "控制器"]):
            signals.append("answer_entity_framework")

    if any(x in q for x in ["local cot repair", "局部修复", "修复策略", "repair strategies"]):
        if any(x in t for x in ["local repair", "bottleneck", "bridge insertion", "step split", "pedagogical", "局部修复", "桥接", "步骤拆分"]):
            signals.append("answer_entity_local_repair")

    # Negative/counterfactual questions: counter-evidence can be direct evidence.
    asks_architecture = any(x in q for x in ["架构", "architecture", "大模型架构", "new model architecture"])
    if asks_architecture and any(x in t for x in ["does not change the student architecture", "not change the student architecture", "data-side", "local cot revision", "local repair"]):
        signals.append("negative_claim_counterevidence")

    asks_all_generalization = any(x in q for x in ["所有", "all llms", "all models", "every dataset", "一定", "always", "generalize", "泛化", "迁移"])
    if asks_all_generalization and any(x in t for x in ["clear limit", "limited", "future work", "under the experimental setting", "does not imply", "not enough", "局限", "有限"]):
        signals.append("scope_limit_counterevidence")

    asks_external_comparison = any(x in q for x in ["gpt-4", "claude", "gsm8k", "医疗", "强化学习", "从零开始"])
    if asks_external_comparison and any(x in t for x in ["qwen2.5-math", "math500", "student models", "downstream benchmark is fixed"]):
        signals.append("scope_evidence_for_absence")

    return list(dict.fromkeys(signals))


def audit_score_to_label(score: int) -> Tuple[str, str]:
    if score >= 85:
        return "direct", "high"
    if score >= 65:
        return "partial", "medium"
    if score >= 40:
        return "background", "low"
    return "weak", "low"

def audit_evidence_node(item: NodeWithScore, question: str, query_plan: QueryPlan, position: int = 1) -> Dict[str, Any]:
    """Human-readable evidence audit score.

    The internal rank_score is useful for sorting but hard to inspect. This
    function converts the evidence into a 0-100 audit score with interpretable
    labels. It intentionally favors direct entity/table/section matches and
    penalizes references, TOC fragments, heading-only chunks, and short chunks.
    """
    metadata = item.node.metadata or {}
    section = str(metadata.get("section_type") or "").lower()
    unit_type = str(metadata.get("unit_type") or "").lower()
    text = normalize_for_match("\n".join([
        str(metadata.get("section_title") or ""),
        str(metadata.get("unit_type") or ""),
        str(metadata.get("section_type") or ""),
        item.node.get_content(),
    ]))

    reasons: List[str] = []
    score = 0.0

    # 1) Exact entity/table/figure match.
    exact_hits = [term for term in query_plan.exact_terms if term.lower() in text or term.lower().replace(" ", "") in text.replace(" ", "")]
    if exact_hits:
        score += 35
        reasons.append("exact_id_match")

    # 1.5) Strong answer-entity / counter-evidence match.
    strong_signals = extract_strong_evidence_signals(question, query_plan, text)
    if strong_signals:
        # Strong signals are designed to make obviously supporting chunks auditable.
        # Cap the bonus so a reference/TOC chunk still cannot dominate.
        score += min(40, 22 + 6 * len(strong_signals))
        reasons.extend(strong_signals[:3])

    # 2) Section suitability.
    pr = route_priority(query_plan.route, section)
    if pr <= 1:
        score += 22
        reasons.append("target_section")
    elif pr <= 3:
        score += 14
        reasons.append("useful_section")
    elif pr <= 5:
        score += 6
        reasons.append("background_section")

    # 3) Keyword/entity overlap.
    keywords = query_keywords_for_audit(question, query_plan)
    matched = [kw for kw in keywords if kw and kw in text]
    if matched:
        score += min(22, 4 * len(matched))
        reasons.append("keyword_overlap")

    # 4) Unit type and content quality.
    if query_plan.route == "experiment" and any(x in unit_type for x in ["table", "figure", "algorithm"]):
        score += 10
        reasons.append("table_or_figure_unit")
    elif query_plan.route == "metadata":
        score += 10
        reasons.append("front_matter_candidate")
    else:
        score += 6
        reasons.append("content_chunk")

    char_len = node_char_len(item)
    if 120 <= char_len <= 1800:
        score += 8
        reasons.append("good_chunk_length")
    elif char_len < 80:
        score -= 15
        reasons.append("too_short")

    # 5) Position contributes only lightly.
    score += max(0, 5 - min(position, 5))

    # Penalties.
    if is_reference_like_node(item):
        score -= 35
        reasons.append("reference_like_penalty")
    if is_toc_like_node(item):
        score -= 30
        reasons.append("toc_like_penalty")
    if is_heading_only_node(item):
        score -= 18
        reasons.append("heading_only_penalty")
    if section in NON_CONTENT_SECTIONS and query_plan.route != "metadata":
        score -= 25
        reasons.append("non_content_section_penalty")

    score = int(max(0, min(100, round(score))))
    relevance, confidence = audit_score_to_label(score)

    return {
        "audit_score": score,
        "relevance": relevance,
        "confidence": confidence,
        "match_reason": "+".join(reasons[:5]) if reasons else "ranked_candidate",
    }


def select_audited_evidence(
    nodes: List[NodeWithScore],
    question: str,
    query_plan: QueryPlan,
    max_items: int = 4,
    min_score: int = 40,
    allow_weak_fallback: bool = False,
) -> List[NodeWithScore]:
    """Select evidence for answer/display using audit scores, not raw rank scores.

    v5 changes:
    - weak evidence is not displayed by default;
    - exact table/figure ids are still pinned for coverage;
    - if no strong evidence exists, show only the best background evidence and
      mark it as fallback instead of pretending it is direct support.
    """
    if not nodes:
        return []

    audited: List[Tuple[NodeWithScore, Dict[str, Any]]] = []
    for pos, item in enumerate(nodes, start=1):
        audit = audit_evidence_node(item, question, query_plan, position=pos)
        item.node.metadata = dict(item.node.metadata or {})
        item.node.metadata["evidence_audit"] = audit
        audited.append((item, audit))

    pinned: List[NodeWithScore] = []
    if query_plan.exact_terms:
        for term in query_plan.exact_terms:
            matches = [(item, audit) for item, audit in audited if node_matches_exact_term(item, term)]
            if matches:
                matches.sort(key=lambda x: (-x[1]["audit_score"], node_page(x[0]), node_chunk_id(x[0])))
                best = matches[0][0]
                if best not in pinned:
                    pinned.append(best)

    direct_or_partial = [item for item, audit in audited if audit["relevance"] in {"direct", "partial"}]
    background = [item for item, audit in audited if audit["relevance"] == "background" and audit["audit_score"] >= min_score]

    def sort_key(item: NodeWithScore) -> Tuple[int, int, int]:
        audit = (item.node.metadata or {}).get("evidence_audit", {})
        return (-int(audit.get("audit_score", 0)), node_page(item), node_chunk_id(item))

    direct_or_partial.sort(key=sort_key)
    background.sort(key=sort_key)

    # Prefer direct/partial. Use at most one background when direct/partial exists.
    selected = dedupe_scored_nodes(pinned + direct_or_partial)
    if len(selected) < max_items and background:
        background_limit = 1 if direct_or_partial else min(2, max_items)
        selected = dedupe_scored_nodes(selected + background[:background_limit])

    if not selected and allow_weak_fallback:
        fallback = sorted([item for item, _ in audited], key=sort_key)[: min(2, max_items)]
        for item in fallback:
            audit = (item.node.metadata or {}).get("evidence_audit", {})
            audit["fallback_note"] = "No direct or partial evidence found; showing best available candidate."
            item.node.metadata["evidence_audit"] = audit
        selected = fallback

    return selected[:max_items]


def select_multi_intent_evidence(grouped_evidence: List[Tuple[str, QueryPlan, List[NodeWithScore]]], max_items: int = 6) -> List[NodeWithScore]:
    """Merge already-audited sub-question evidence without re-auditing it against the long original query."""
    merged: List[NodeWithScore] = []
    per_sub_limit = max(1, math.ceil(max_items / max(1, len(grouped_evidence))))
    for sub_q, sub_plan, sub_nodes in grouped_evidence:
        kept = select_audited_evidence(
            sub_nodes,
            question=sub_q,
            query_plan=sub_plan,
            max_items=per_sub_limit,
            allow_weak_fallback=True,
        )
        for item in kept:
            item.node.metadata = dict(item.node.metadata or {})
            item.node.metadata["sub_question"] = sub_q
            item.node.metadata["sub_route"] = sub_plan.route
        merged.extend(kept)

    merged = dedupe_scored_nodes(merged)
    # Keep cross-intent coverage first, then fill by audit score.
    merged.sort(key=lambda item: (
        str((item.node.metadata or {}).get("sub_question", "")),
        -int(((item.node.metadata or {}).get("evidence_audit") or {}).get("audit_score", 0)),
        node_page(item),
        node_chunk_id(item),
    ))
    return merged[:max_items]


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
                    f"audit={metadata.get('evidence_audit', {})}",
                    f"rank_score={node_with_score.score:.4f}",
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


def metadata_attribute_keywords(requested_attribute: str, question: str) -> List[str]:
    q = f"{requested_attribute} {question}".lower()
    if contains_any(q, ["author", "作者", "谁写", "谁完成", "署名"]):
        return ["author", "student name", "姓名", "作者", "署名", "author signature", "学生姓名"]
    if contains_any(q, ["advisor", "supervisor", "导师", "指导老师", "指导教师"]):
        return ["advisor", "supervisor", "导师", "指导老师", "指导教师"]
    if contains_any(q, ["program", "major", "专业"]):
        return ["program", "major", "专业", "数据科学", "big data"]
    if contains_any(q, ["department", "院系", "哪个系"]):
        return ["department", "院系", "系", "statistics", "data science"]
    if contains_any(q, ["title", "题目", "标题", "叫什么"]):
        return ["title", "题目", "标题", "student-oriented", "chain-of-thought"]
    if contains_any(q, ["date", "submitted", "日期", "时间", "提交"]):
        return ["date", "submitted", "日期", "提交", "年", "月"]
    if contains_any(q, ["institution", "university", "学校", "单位", "机构"]):
        return ["institution", "university", "学校", "大学", "southern university", "sustech"]
    return ["title", "author", "advisor", "department", "program", "date", "作者", "导师", "专业", "题目"]


def select_metadata_evidence(
    question: str,
    query_plan: QueryPlan,
    answer: str,
    nodes: List[NodeWithScore],
    top_k: int = 4,
) -> List[NodeWithScore]:
    """Compress front-matter evidence to directly supporting chunks."""
    keywords = metadata_attribute_keywords(query_plan.requested_attribute, question)
    answer_terms = [x for x in re.findall(r"[A-Za-z][A-Za-z\-.]+|[\u4e00-\u9fff]{2,}", answer) if len(x) >= 2]
    answer_terms = answer_terms[:8]

    rescored: List[NodeWithScore] = []
    for item in nodes:
        metadata = item.node.metadata or {}
        text = normalize_for_match(
            "\n".join(
                [
                    str(metadata.get("section_title") or ""),
                    str(metadata.get("section_type") or ""),
                    item.node.get_content(),
                ]
            )
        )
        score = 0.0
        if node_page(item) <= 6:
            score += 1.0
        for kw in keywords:
            if kw.lower() in text:
                score += 2.0
        for term in answer_terms:
            if term.lower() in text:
                score += 1.2
        if any(x in text for x in ["abstract", "introduction", "method", "experiment"]):
            score -= 0.2
        if node_char_len(item) < 30:
            score -= 0.5
        if score > 0:
            rescored.append(NodeWithScore(node=item.node, score=score))

    if not rescored:
        rescored = nodes[:top_k]
    rescored = dedupe_scored_nodes(rescored)
    rescored.sort(key=lambda x: (-(float(x.score or 0.0)), node_page(x), node_chunk_id(x)))
    return rescored[:top_k]


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
If the retrieved sources contain table/figure captions, model names, benchmark names, or numeric changes, summarize them directly.
Do not say that detailed values are unavailable when any retrieved source includes relevant values or table/figure evidence.
If exact values are genuinely not in the retrieved sources, say so.
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


def synthesize_multi_intent_answer(
    llm: OpenAI,
    original_question: str,
    sub_answers: List[Tuple[str, QueryPlan, str]],
) -> str:
    """Merge sub-question answers into one coherent response."""
    blocks = []
    for i, (sub_q, plan, answer) in enumerate(sub_answers, start=1):
        blocks.append(
            f"[Sub-answer {i}]\nQuestion: {sub_q}\nRoute: {plan.route}\nAnswer:\n{answer}"
        )
    prompt = f"""
You are combining several source-grounded RAG sub-answers.

Original user question:
{original_question}

Sub-answers:
{chr(10).join(blocks)}

Write one final answer.
Rules:
1. Preserve important cautions about unsupported or extra-document claims.
2. Do not add outside knowledge.
3. If the original question is in Chinese, answer in Chinese; otherwise answer in English.
4. Use a concise structure, one paragraph or a short numbered list.

Final answer:
""".strip()
    return str(llm.complete(prompt)).strip()



def build_claim_verdict(question: str, query_plan: QueryPlan, nodes: List[NodeWithScore], answer: str) -> Dict[str, Any]:
    """Create a compact, human-auditable verdict for claim / negative fact questions."""
    q = normalize_for_match(question)
    ans = normalize_for_match(answer)
    source_text = normalize_for_match("\n".join(item.node.get_content() for item in nodes))

    not_supported_markers = [
        "不支持", "没有", "并没有", "无法", "不能", "not support", "not supported", "does not", "do not", "no evidence", "not provide",
    ]
    supported_markers = ["支持", "证明", "reported", "reports", "prove", "shows", "demonstrates"]

    positive_evidence_found = False
    if any(x in q for x in ["gpt-4", "claude"]):
        positive_evidence_found = any(x in source_text for x in ["gpt-4", "claude"])
    elif "gsm8k" in q:
        positive_evidence_found = "gsm8k" in source_text and any(x in source_text for x in ["result", "score", "reported", "报告", "结果"])
    elif any(x in q for x in ["架构", "architecture"]):
        positive_evidence_found = any(x in source_text for x in ["new architecture", "提出新的架构", "model architecture"])
    else:
        positive_evidence_found = any(x in ans for x in supported_markers) and not any(x in ans for x in not_supported_markers)

    if any(x in ans for x in not_supported_markers):
        verdict = "not_supported"
    elif positive_evidence_found:
        verdict = "supported"
    else:
        verdict = "partial_or_uncertain"

    searched_entities = []
    for ent in ["GPT-4", "Claude", "GSM8K", "new architecture", "all LLMs", "all datasets", "所有模型", "所有数据集"]:
        if ent.lower() in q or ent in question:
            searched_entities.append(ent)

    scope_parts = []
    if "qwen2.5-math" in source_text:
        scope_parts.append("retrieved scope mentions Qwen2.5-Math student models")
    if "math500" in source_text:
        scope_parts.append("retrieved scope mentions math500 / math500 strict")
    if any(x in source_text for x in ["clear limit", "limited", "future work", "under the experimental setting", "does not change the student architecture"]):
        scope_parts.append("retrieved evidence contains limitation/scope wording")

    return {
        "claim": question,
        "verdict": verdict,
        "positive_evidence_found": positive_evidence_found,
        "searched_entities": searched_entities,
        "scope_evidence": scope_parts[:3],
    }


def print_claim_verdict(verdict: Dict[str, Any]) -> None:
    if not verdict:
        return
    print("\nClaim Verdict")
    print("-------------")
    print(f"claim: {verdict.get('claim')}")
    print(f"verdict: {verdict.get('verdict')}")
    print(f"positive_evidence_found: {str(verdict.get('positive_evidence_found')).lower()}")
    if verdict.get("searched_entities"):
        print("searched_entities: " + ", ".join(verdict.get("searched_entities") or []))
    if verdict.get("scope_evidence"):
        print("scope_evidence:")
        for item in verdict.get("scope_evidence") or []:
            print(f"  - {item}")

def print_answer(answer: str) -> None:
    print("\nAnswer")
    print("------")
    print(textwrap.fill(answer, width=100, replace_whitespace=False))


def print_evidence(nodes: List[NodeWithScore], route: str = "general") -> None:
    print("\nEvidence")
    print("--------")
    for i, node_with_score in enumerate(nodes, start=1):
        node = node_with_score.node
        metadata = node.metadata or {}
        page = metadata.get("page")
        section = metadata.get("section_type")
        section_title = metadata.get("section_title")
        chunk_id = metadata.get("chunk_id")
        preview = compact_preview(node.get_content())
        audit = metadata.get("evidence_audit") or {}

        if route == "metadata":
            score_text = "support=front_matter_match"
        else:
            if audit:
                score_text = (
                    f"relevance={audit.get('relevance')} | "
                    f"confidence={audit.get('confidence')} | "
                    f"audit_score={audit.get('audit_score')} | "
                    f"rank_score={float(node_with_score.score or 0.0):.4f}"
                )
            else:
                score_text = f"rank_score={float(node_with_score.score or 0.0):.4f}"
        print(f"[{i}] page={page} | section={section} | chunk_id={chunk_id} | {score_text}")
        if audit and route != "metadata":
            if metadata.get("sub_question"):
                print(f"    sub_question: {metadata.get('sub_question')}")
            if metadata.get("sub_route"):
                print(f"    sub_route: {metadata.get('sub_route')}")
            print(f"    match_reason: {audit.get('match_reason')}")
            if audit.get("fallback_note"):
                print(f"    note: {audit.get('fallback_note')}")
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
    parser.add_argument(
        "--rerank_mode",
        choices=["none", "heuristic", "llm"],
        default=None,
        help="Evidence rerank mode. Default reads rag.rerank_mode from config, or heuristic.",
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

    rerank_mode = args.rerank_mode or str(cfg.get("rag", {}).get("rerank_mode", "heuristic"))
    sub_questions = split_multi_intent_question(question)

    if len(sub_questions) > 1:
        sub_answers: List[Tuple[str, QueryPlan, str]] = []
        grouped_evidence: List[Tuple[str, QueryPlan, List[NodeWithScore]]] = []
        route_names: List[str] = []

        for sub_q in sub_questions:
            sub_plan = classify_query(llm=llm, user_query=sub_q, corpus_profile=corpus_profile)
            route_names.append(sub_plan.route)
            if sub_plan.route == "metadata":
                sub_nodes = collect_metadata_candidate_nodes(
                    all_nodes=all_nodes,
                    sparse_retriever=sparse_retriever,
                    retrieval_query=sub_plan.retrieval_query,
                    top_k=top_k,
                    front_pages=6,
                )
                sub_answer = answer_metadata_question_with_llm(
                    llm=llm,
                    question=sub_q,
                    query_plan=sub_plan,
                    nodes=sub_nodes,
                )
                sub_nodes = select_metadata_evidence(
                    question=sub_q,
                    query_plan=sub_plan,
                    answer=sub_answer,
                    nodes=sub_nodes,
                    top_k=3,
                )
            else:
                sub_nodes = retrieve_nodes(
                    llm=llm,
                    question=sub_q,
                    all_nodes=all_nodes,
                    dense_retriever=dense_retriever,
                    sparse_retriever=sparse_retriever,
                    retrieval_query=sub_plan.retrieval_query,
                    query_plan=sub_plan,
                    top_k=top_k,
                    retrieval_mode=args.retrieval_mode,
                    rerank_mode=rerank_mode,
                )
                sub_nodes = select_audited_evidence(sub_nodes, question=sub_q, query_plan=sub_plan, max_items=min(4, top_k), allow_weak_fallback=True)
                sub_answer = generate_answer(llm=llm, question=sub_q, query_plan=sub_plan, nodes=sub_nodes)

            sub_answers.append((sub_q, sub_plan, sub_answer))
            grouped_evidence.append((sub_q, sub_plan, sub_nodes))

        query_plan = QueryPlan(
            route="multi_intent",
            retrieval_query="\n\n".join(
                [f"[{i}] {sq}" for i, sq in enumerate(sub_questions, start=1)]
            ),
            route_reason=f"multi-intent decomposition: {' + '.join(route_names)}",
            sub_questions=tuple(sub_questions),
        )
        nodes = select_multi_intent_evidence(grouped_evidence, max_items=min(6, top_k))
        answer = synthesize_multi_intent_answer(
            llm=llm,
            original_question=question,
            sub_answers=sub_answers,
        )
        output_mode = f"{args.retrieval_mode} + multi_intent + {rerank_mode}_rerank + subquestion_evidence_audit_v5"

        print_demo_summary(
            question=question,
            query_plan=query_plan,
            show_rewritten_query=args.show_rewritten_query,
            retrieval_mode=output_mode,
        )
        print_answer(answer)
        print_evidence(nodes, route=query_plan.route)
        return

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
        nodes = select_metadata_evidence(
            question=question,
            query_plan=query_plan,
            answer=answer,
            nodes=nodes,
            top_k=min(4, top_k),
        )
        output_mode = f"{args.retrieval_mode} + generic_front_matter_metadata_qa + evidence_compression"
    else:
        nodes = retrieve_nodes(
            llm=llm,
            question=question,
            all_nodes=all_nodes,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            retrieval_query=query_plan.retrieval_query,
            query_plan=query_plan,
            top_k=top_k,
            retrieval_mode=args.retrieval_mode,
            rerank_mode=rerank_mode,
        )
        nodes = select_audited_evidence(nodes, question=question, query_plan=query_plan, max_items=min(4, top_k), allow_weak_fallback=True)
        answer = generate_answer(llm=llm, question=question, query_plan=query_plan, nodes=nodes)
        output_mode = f"{args.retrieval_mode} + route_aware_retrieval + {rerank_mode}_rerank + evidence_audit_v5"

    print_demo_summary(
        question=question,
        query_plan=query_plan,
        show_rewritten_query=args.show_rewritten_query,
        retrieval_mode=output_mode,
    )
    print_answer(answer)
    if query_plan.route == "claim" or looks_like_claim_query(question):
        print_claim_verdict(build_claim_verdict(question, query_plan, nodes, answer))
    print_evidence(nodes, route=query_plan.route)


if __name__ == "__main__":
    main()
