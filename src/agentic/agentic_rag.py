#src/agentic/agentic_rag.py
"""
Corpus-aware Agentic RAG workflow.

This version removes thesis-specific hardcoded entities from prompts and checks.
It keeps the useful agentic workflow:

1. Query rewriting
2. Retrieval
3. Evidence sufficiency checking
4. Retry retrieval when evidence is insufficient
5. Neighbor-context expansion with previous_chunk_id / next_chunk_id
6. Source-grounded answer generation
7. Citation audit
8. Answer revision

Document-specific information should live in configs/corpus_profile.yaml, not in code.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


DEFAULT_PROFILE: Dict[str, Any] = {
    "corpus": {
        "name": "research_corpus",
        "title": "Research Document Corpus",
        "description": "A source-grounded RAG corpus over research or technical documents.",
        "domain": "research and technical document QA",
        "topics": [
            "research problem",
            "method",
            "experimental setup",
            "results",
            "tables and figures",
            "limitations and future work",
            "claim verification",
        ],
        "optional_keywords": [
            "abstract",
            "introduction",
            "related work",
            "method",
            "experiment",
            "evaluation",
            "results",
            "discussion",
            "conclusion",
            "limitation",
            "future work",
            "table",
            "figure",
        ],
        "entity_types": {
            "benchmarks": [],
            "models": [],
            "datasets": [],
            "methods": [],
            "metrics": [],
        },
    }
}


def load_corpus_profile(profile_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load corpus profile. Falls back to a generic profile when the file is absent."""
    if not profile_path:
        return DEFAULT_PROFILE

    path = Path(profile_path)
    if not path.exists():
        return DEFAULT_PROFILE

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if "corpus" not in data:
        return DEFAULT_PROFILE

    merged = json.loads(json.dumps(DEFAULT_PROFILE))
    merged["corpus"].update(data.get("corpus", {}))

    default_entity_types = DEFAULT_PROFILE["corpus"]["entity_types"]
    user_entity_types = data.get("corpus", {}).get("entity_types", {}) or {}
    merged["corpus"]["entity_types"] = {**default_entity_types, **user_entity_types}

    return merged


class AgenticRAG:
    def __init__(
        self,
        llm: Any,
        index: Any,
        storage_context: Any,
        top_k: int = 8,
        max_retry: int = 1,
        max_sources_for_answer: int = 8,
        max_sources_for_judge: int = 6,
        corpus_profile: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.llm = llm
        self.index = index
        self.storage_context = storage_context
        self.top_k = top_k
        self.max_retry = max_retry
        self.max_sources_for_answer = max_sources_for_answer
        self.max_sources_for_judge = max_sources_for_judge
        self.corpus_profile = corpus_profile or DEFAULT_PROFILE

        self.retriever = index.as_retriever(similarity_top_k=top_k)
        self.chunk_lookup = self._build_chunk_lookup()

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------

    def _build_chunk_lookup(self) -> Dict[int, Any]:
        lookup: Dict[int, Any] = {}
        docs = getattr(self.storage_context.docstore, "docs", {})

        for _, node in docs.items():
            metadata = node.metadata or {}
            chunk_id = metadata.get("chunk_id")
            if chunk_id is None:
                continue
            try:
                lookup[int(chunk_id)] = node
            except Exception:
                continue

        return lookup

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _extract_json_object(text: str) -> Dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Could not find JSON object in response:\n{text}")

        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse JSON object:\n{candidate}") from e

    @staticmethod
    def _clamp_score(value: Any, default: float = 0.0) -> float:
        try:
            v = float(value)
        except Exception:
            return default
        return max(0.0, min(5.0, v))

    @staticmethod
    def _has_negative_marker(text: str) -> bool:
        negative_markers = [
            "not",
            "does not",
            "do not",
            "cannot",
            "no evidence",
            "not supported",
            "does not support",
            "not report",
            "does not report",
            "not compare",
            "does not compare",
            "not prove",
            "does not prove",
            "unsupported",
            "insufficient evidence",
            "没有",
            "不支持",
            "不能",
            "无法",
            "未",
            "没有证据",
        ]
        lower = text.lower()
        return any(marker in lower for marker in negative_markers)

    def _corpus(self) -> Dict[str, Any]:
        return self.corpus_profile.get("corpus", {}) or {}

    def _profile_context_text(self) -> str:
        corpus = self._corpus()
        title = corpus.get("title", "")
        description = corpus.get("description", "")
        domain = corpus.get("domain", "")
        topics = corpus.get("topics", []) or []
        optional_keywords = corpus.get("optional_keywords", []) or []
        entity_types = corpus.get("entity_types", {}) or {}

        lines = [
            f"Corpus title: {title}",
            f"Corpus description: {description}",
            f"Corpus domain: {domain}",
        ]

        if topics:
            lines.append("Important corpus topics:")
            lines.extend(f"- {x}" for x in topics)

        if optional_keywords:
            lines.append("Useful generic retrieval keywords:")
            lines.extend(f"- {x}" for x in optional_keywords)

        filled_entities = {
            k: v for k, v in entity_types.items() if isinstance(v, list) and len(v) > 0
        }
        if filled_entities:
            lines.append("Known corpus entities from profile. Use them only when relevant to the user query:")
            for entity_type, values in filled_entities.items():
                lines.append(f"- {entity_type}: {', '.join(str(x) for x in values)}")

        return "\n".join(lines)

    def _profile_entity_map(self) -> Dict[str, List[str]]:
        entity_types = self._corpus().get("entity_types", {}) or {}
        result: Dict[str, List[str]] = {}
        for entity_type, values in entity_types.items():
            if not isinstance(values, list):
                continue
            cleaned = [str(v).strip() for v in values if str(v).strip()]
            result[str(entity_type)] = cleaned
        return result

    def _detect_profile_entities(self, text: str) -> Dict[str, List[str]]:
        lower = text.lower()
        detected: Dict[str, List[str]] = {}
        for entity_type, entities in self._profile_entity_map().items():
            hits = []
            for entity in entities:
                if entity.lower() in lower:
                    hits.append(entity)
            detected[entity_type] = hits
        return detected

    def _generic_entity_mismatch_flags(
        self,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Generic deterministic guardrail using optional profile entities.

        It does not hardcode any concrete model, benchmark, or method names.
        It only checks whether evidence about one known entity type is being used
        to answer a question about a different entity of the same type.
        """
        flags: List[str] = []
        if self._has_negative_marker(answer):
            return flags

        source_text = "\n".join(str(s.get("text", "")) for s in sources)
        q_entities = self._detect_profile_entities(question)
        a_entities = self._detect_profile_entities(answer)
        s_entities = self._detect_profile_entities(source_text)

        for entity_type in self._profile_entity_map().keys():
            q_hits = set(q_entities.get(entity_type, []))
            a_hits = set(a_entities.get(entity_type, []))
            s_hits = set(s_entities.get(entity_type, []))

            if not q_hits:
                continue

            missing_in_sources = q_hits - s_hits
            answer_uses_other = a_hits - q_hits

            if missing_in_sources and answer_uses_other:
                flags.append(
                    f"Possible {entity_type} mismatch: the question asks about "
                    f"{sorted(missing_in_sources)}, but the answer uses other {entity_type} "
                    f"{sorted(answer_uses_other)} without direct source support."
                )

        universal_patterns = [
            "all models",
            "all llms",
            "all benchmarks",
            "all datasets",
            "every setting",
            "always",
            "所有模型",
            "所有基准",
            "所有数据集",
            "所有设置",
            "一定",
            "总是",
        ]
        q_lower = question.lower()
        a_lower = answer.lower()
        if any(p in q_lower or p in a_lower for p in universal_patterns):
            if not self._has_negative_marker(answer):
                flags.append(
                    "Possible overgeneralization: universal claims require explicit evidence across all stated settings."
                )

        return flags

    def _node_to_source_record(
        self,
        node: Any,
        score: Optional[float] = None,
        rank: Optional[int] = None,
        relation: str = "retrieved",
        max_preview_chars: int = 900,
    ) -> Dict[str, Any]:
        metadata = node.metadata or {}
        text = node.get_content()

        return {
            "rank": rank,
            "score": self._safe_float(score),
            "relation": relation,
            "metadata": {
                "source": metadata.get("source"),
                "doc_id": metadata.get("doc_id"),
                "title": metadata.get("title"),
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

    def _source_nodes_to_records(self, source_nodes: List[Any]) -> List[Dict[str, Any]]:
        records = []
        for rank, source_node in enumerate(source_nodes, start=1):
            records.append(
                self._node_to_source_record(
                    node=source_node.node,
                    score=source_node.score,
                    rank=rank,
                    relation="retrieved",
                )
            )
        return records

    @staticmethod
    def _source_key(source: Dict[str, Any]) -> Tuple[str, str]:
        metadata = source.get("metadata", {})
        chunk_id = metadata.get("chunk_id")
        source_path = metadata.get("source") or metadata.get("doc_id") or ""
        if chunk_id is not None:
            return (str(source_path), str(chunk_id))
        return (str(source_path), source.get("text", "")[:200])

    def _merge_sources(self, source_groups: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        merged = []
        seen = set()
        for group in source_groups:
            for source in group:
                key = self._source_key(source)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(source)

        for i, source in enumerate(merged, start=1):
            source["rank"] = i
        return merged

    @staticmethod
    def _format_sources(
        sources: List[Dict[str, Any]],
        max_sources: int = 8,
        max_chars_each: int = 1300,
    ) -> str:
        parts = []
        for source in sources[:max_sources]:
            metadata = source.get("metadata", {})
            text = source.get("text", "").replace("\n", " ").strip()
            if len(text) > max_chars_each:
                text = text[:max_chars_each] + "..."

            parts.append(
                f"[Source {source.get('rank')} | relation={source.get('relation')}]\n"
                f"source={metadata.get('source')}, doc_id={metadata.get('doc_id')}, "
                f"page={metadata.get('page')}, page_end={metadata.get('page_end')}, "
                f"section_type={metadata.get('section_type')}, "
                f"section_title={metadata.get('section_title')}, "
                f"chunk_id={metadata.get('chunk_id')}, "
                f"unit_type={metadata.get('unit_type')}\n"
                f"text={text}"
            )
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Tool 1: Query rewrite
    # ------------------------------------------------------------------

    def rewrite_query(self, question: str) -> str:
        profile_context = self._profile_context_text()
        prompt = f"""
You are a query rewriting module for a source-grounded RAG system.

Your job is to rewrite the user query into a retrieval query that is easier to match against document chunks.
Use the corpus profile only as optional context. Do not force irrelevant profile terms into the query.

Corpus profile:
{profile_context}

Strict rules:
1. Do not answer the question.
2. Preserve the original intent.
3. Do not add unsupported assumptions.
4. Add only closely related terms that help retrieval.
5. For numerical, table, figure, model, dataset, benchmark, or metric questions, include the relevant generic words such as table, figure, result, metric, dataset, benchmark, model, experiment, comparison, evaluation.
6. For limitation or future-work questions, include terms such as limitation, future work, generalization, scope, conclusion.
7. If the query is in Chinese, keep important Chinese terms and add useful English equivalents when helpful.
8. Return only the rewritten retrieval query, no explanation.

User query:
{question}

Rewritten retrieval query:
""".strip()

        response = self.llm.complete(prompt)
        rewritten = str(response).strip()
        if not rewritten:
            return question
        return f"{question}\n{rewritten}"

    # ------------------------------------------------------------------
    # Tool 2: Search corpus
    # ------------------------------------------------------------------

    def search_corpus(self, query: str) -> List[Dict[str, Any]]:
        source_nodes = self.retriever.retrieve(query)
        return self._source_nodes_to_records(source_nodes)

    # Backward-compatible alias for your old scripts.
    def search_thesis(self, query: str) -> List[Dict[str, Any]]:
        return self.search_corpus(query)

    # ------------------------------------------------------------------
    # Tool 3: Evidence sufficiency check
    # ------------------------------------------------------------------

    def check_evidence_sufficiency(
        self,
        question: str,
        sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        source_text = self._format_sources(
            sources=sources,
            max_sources=self.max_sources_for_judge,
            max_chars_each=1200,
        )

        prompt = f"""
You are an evidence sufficiency checker for an Agentic RAG system.

Decide whether the retrieved sources contain enough evidence to answer the question.

Question:
{question}

Retrieved sources:
{source_text}

Evaluation rules:
1. Judge only whether the sources contain enough evidence to answer the question.
2. Do not answer the question.
3. If the question asks for exact numbers, tables, figures, model names, dataset names, benchmark names, metrics, or comparisons, the sources must contain those exact details.
4. If the question asks to compare two objects, sources should contain evidence for both objects.
5. If the question asks why/how, sources should include explanatory discussion, not only a short caption or isolated phrase.
6. If the question asks whether a claim is supported, the sources must directly support the same claim, not only a related claim.
7. Return valid JSON only.

Return this JSON object:
{{
  "sufficient": true or false,
  "confidence": 0-5,
  "reason": "brief reason",
  "missing_aspects": ["missing aspect 1", "missing aspect 2"],
  "suggested_retry_query": "a better retrieval query if sources are insufficient, otherwise empty string"
}}
""".strip()

        raw = str(self.llm.complete(prompt)).strip()
        try:
            parsed = self._extract_json_object(raw)
        except Exception as e:
            parsed = {
                "sufficient": True,
                "confidence": 2,
                "reason": f"Failed to parse evidence check JSON, defaulting to sufficient. Error: {e}",
                "missing_aspects": [],
                "suggested_retry_query": "",
                "raw_response": raw,
            }

        if not isinstance(parsed.get("sufficient"), bool):
            parsed["sufficient"] = bool(parsed.get("sufficient"))
        parsed["confidence"] = self._clamp_score(parsed.get("confidence"), default=2.0)
        if not isinstance(parsed.get("missing_aspects"), list):
            parsed["missing_aspects"] = []
        parsed.setdefault("suggested_retry_query", "")
        parsed.setdefault("reason", "")
        return parsed

    # ------------------------------------------------------------------
    # Tool 4: Retry retrieval
    # ------------------------------------------------------------------

    def build_retry_query(self, question: str, evidence_check: Dict[str, Any]) -> str:
        suggested_retry_query = str(evidence_check.get("suggested_retry_query", "")).strip()
        if suggested_retry_query:
            return f"{question}\n{suggested_retry_query}"

        missing_aspects = evidence_check.get("missing_aspects", [])
        missing_text = "; ".join(str(x) for x in missing_aspects)
        generic_terms = ", ".join(str(x) for x in self._corpus().get("optional_keywords", [])[:16])

        return (
            f"{question}\n"
            f"Missing aspects to retrieve: {missing_text}\n"
            f"Use relevant corpus keywords only if useful: {generic_terms}."
        )

    # ------------------------------------------------------------------
    # Tool 5: Neighbor context expansion
    # ------------------------------------------------------------------

    def expand_neighbor_context(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        expanded_groups = [sources]
        neighbor_records = []

        def looks_incomplete(text: str) -> bool:
            text = text.strip()
            if not text:
                return False
            incomplete_endings = (":", ";", ",", "and", "or", "including", "such as")
            lower = text.lower()
            if lower.endswith(incomplete_endings):
                return True
            if len(text) < 220 and not text.endswith((".", "?", "!", "。", "？", "！")):
                return True
            return False

        for source in sources[: min(self.max_sources_for_answer, 5)]:
            metadata = source.get("metadata", {})
            unit_type = metadata.get("unit_type")
            char_len = metadata.get("char_len") or 0
            text = source.get("text", "")

            try:
                char_len_int = int(char_len)
            except Exception:
                char_len_int = len(text)

            should_expand = (
                unit_type in {"table", "figure"}
                or char_len_int <= 220
                or looks_incomplete(text)
            )
            if not should_expand:
                continue

            for neighbor_field, relation in [
                ("previous_chunk_id", "previous_neighbor"),
                ("next_chunk_id", "next_neighbor"),
            ]:
                neighbor_id = metadata.get(neighbor_field)
                if neighbor_id is None:
                    continue
                try:
                    neighbor_id_int = int(neighbor_id)
                except Exception:
                    continue
                neighbor_node = self.chunk_lookup.get(neighbor_id_int)
                if neighbor_node is None:
                    continue
                neighbor_records.append(
                    self._node_to_source_record(
                        node=neighbor_node,
                        score=None,
                        rank=None,
                        relation=relation,
                    )
                )

        if neighbor_records:
            expanded_groups.append(neighbor_records)
        return self._merge_sources(expanded_groups)

    # ------------------------------------------------------------------
    # Tool 6: Generate grounded answer
    # ------------------------------------------------------------------

    def generate_answer(self, question: str, sources: List[Dict[str, Any]]) -> str:
        source_text = self._format_sources(
            sources=sources,
            max_sources=self.max_sources_for_answer,
            max_chars_each=1400,
        )

        prompt = f"""
You are a source-grounded QA assistant.

You must answer the question using only the retrieved sources.

Question:
{question}

Retrieved sources:
{source_text}

Core rules:
1. Use only information directly supported by the retrieved sources.
2. Do not invent numbers, model names, dataset names, benchmark names, metrics, baselines, comparisons, mechanisms, or conclusions.
3. If the sources do not explicitly support a claim, clearly say that the retrieved sources do not support it.
4. Do not answer "yes" just because the sources contain related terms. Related evidence is not the same as direct evidence.
5. If the question asks whether a document "reports", "proves", "compares", "claims", "shows", "supports", or "uses" something, treat it as claim verification.
6. For claim-verification questions, first decide whether the retrieved sources directly support the same claim.
7. If unsupported, answer cautiously and explain what the sources do support instead.

Generic mismatch rules:
1. Evidence about one benchmark, dataset, model, method, metric, table, figure, or setting cannot support a claim about another unless the sources explicitly connect them.
2. Related work discussion does not count as an experimental comparison unless the sources explicitly report that comparison.
3. Evidence of improvement in a limited setting does not support universal claims such as "always", "all models", "all benchmarks", or "all datasets".
4. Method, training, data, retrieval, and architecture are different mechanisms. Do not merge them unless the sources explicitly do so.

Answer style:
1. Be direct.
2. If supported, answer with source-grounded details.
3. If unsupported, explicitly say it is unsupported by the retrieved sources.
4. Mention source pages naturally when useful.
5. Keep the answer concise but complete.

Answer:
""".strip()

        response = self.llm.complete(prompt)
        return str(response).strip()

    # ------------------------------------------------------------------
    # Tool 7: Citation audit
    # ------------------------------------------------------------------

    def audit_answer(self, question: str, answer: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        source_text = self._format_sources(
            sources=sources,
            max_sources=self.max_sources_for_judge,
            max_chars_each=1300,
        )

        prompt = f"""
You are a strict but fair citation audit judge for a source-grounded RAG system.

Your job is to verify whether the answer is faithful to the retrieved sources.

Question:
{question}

Answer:
{answer}

Retrieved sources:
{source_text}

Evaluate the answer along these dimensions:
1. correctness: Does it correctly address the question?
2. completeness: Does it cover the necessary points without requiring unnecessary details?
3. groundedness: Are the important factual claims grounded in the sources?
4. citation_support: Do the retrieved sources directly support the main claims?
5. hallucination_risk: Does the answer introduce unsupported or overgeneralized claims?

Claim-verification rules:
1. If the question asks whether something is reported, proven, compared, claimed, shown, supported, or used, the answer must make a clear support / not-supported judgment.
2. If the sources do not directly support the claim, a cautious negative answer should receive high correctness and groundedness.
3. Evidence about one entity cannot support a different entity unless the sources explicitly connect them.
4. Universal claims require explicit broad evidence.

Return valid JSON only:
{{
  "question_type": "direct_qa" | "claim_verification" | "comparison" | "table_or_figure" | "limitation" | "unknown",
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
  "missing_points": ["missing point 1", "missing point 2"],
  "directly_supported": true or false,
  "requires_cautious_answer": true or false,
  "revision_needed": true or false
}}
""".strip()

        raw = str(self.llm.complete(prompt)).strip()
        try:
            parsed = self._extract_json_object(raw)
        except Exception as e:
            parsed = {
                "question_type": "unknown",
                "correctness": 0,
                "completeness": 0,
                "groundedness": 0,
                "citation_support": 0,
                "hallucination_risk": 5,
                "overall_score": 0,
                "verdict": "poor",
                "reason": f"Audit JSON parsing failed: {e}",
                "supported_claims": [],
                "unsupported_claims": [],
                "missing_points": [],
                "directly_supported": False,
                "requires_cautious_answer": True,
                "revision_needed": True,
                "raw_response": raw,
            }

        for key in [
            "correctness",
            "completeness",
            "groundedness",
            "citation_support",
            "hallucination_risk",
            "overall_score",
        ]:
            default = 5.0 if key == "hallucination_risk" else 0.0
            parsed[key] = self._clamp_score(parsed.get(key), default=default)

        for key in ["supported_claims", "unsupported_claims", "missing_points"]:
            if not isinstance(parsed.get(key), list):
                parsed[key] = []

        parsed.setdefault("question_type", "unknown")
        parsed.setdefault("verdict", "partial")
        parsed.setdefault("reason", "")

        if not isinstance(parsed.get("directly_supported"), bool):
            parsed["directly_supported"] = parsed["citation_support"] >= 4
        if not isinstance(parsed.get("requires_cautious_answer"), bool):
            parsed["requires_cautious_answer"] = False

        mismatch_flags = self._generic_entity_mismatch_flags(question, answer, sources)
        if mismatch_flags:
            parsed["unsupported_claims"].extend(mismatch_flags)
            parsed["citation_support"] = min(parsed["citation_support"], 2.0)
            parsed["groundedness"] = min(parsed["groundedness"], 2.0)
            parsed["hallucination_risk"] = max(parsed["hallucination_risk"], 3.0)
            parsed["overall_score"] = min(parsed["overall_score"], 2.0)
            parsed["verdict"] = "poor"
            parsed["revision_needed"] = True
            parsed["requires_cautious_answer"] = True
            parsed["directly_supported"] = False
            parsed["reason"] = (
                str(parsed.get("reason", ""))
                + " Generic entity mismatch check flagged unsupported claim risk: "
                + "; ".join(mismatch_flags)
            ).strip()

        if not isinstance(parsed.get("revision_needed"), bool):
            parsed["revision_needed"] = (
                parsed["citation_support"] < 4
                or parsed["hallucination_risk"] >= 2
                or len(parsed["unsupported_claims"]) > 0
            )

        return parsed

    # ------------------------------------------------------------------
    # Tool 8: Answer revision
    # ------------------------------------------------------------------

    def revise_answer(
        self,
        question: str,
        original_answer: str,
        audit: Dict[str, Any],
        sources: List[Dict[str, Any]],
    ) -> str:
        source_text = self._format_sources(
            sources=sources,
            max_sources=self.max_sources_for_answer,
            max_chars_each=1400,
        )

        unsupported_claims = audit.get("unsupported_claims", [])
        missing_points = audit.get("missing_points", [])

        prompt = f"""
You are revising a source-grounded answer after citation audit.

Question:
{question}

Original answer:
{original_answer}

Audit unsupported claims:
{json.dumps(unsupported_claims, ensure_ascii=False)}

Audit missing points:
{json.dumps(missing_points, ensure_ascii=False)}

Retrieved sources:
{source_text}

Revision rules:
1. Remove or soften unsupported claims.
2. Keep only claims directly supported by the sources.
3. If the sources do not support the requested claim, clearly say so.
4. If evidence is partial, say what is supported and what is not supported.
5. Do not invent new evidence.
6. Keep the answer concise.

Revised answer:
""".strip()

        response = self.llm.complete(prompt)
        return str(response).strip()

    # ------------------------------------------------------------------
    # Main workflow
    # ------------------------------------------------------------------

    def answer(self, question: str) -> Dict[str, Any]:
        rewritten_query = self.rewrite_query(question)
        initial_sources = self.search_corpus(rewritten_query)

        all_source_groups = [initial_sources]
        retry_queries: List[str] = []
        evidence_check = self.check_evidence_sufficiency(question, initial_sources)

        current_sources = initial_sources
        retry_count = 0

        while not evidence_check.get("sufficient", True) and retry_count < self.max_retry:
            retry_query = self.build_retry_query(question, evidence_check)
            retry_queries.append(retry_query)
            retry_sources = self.search_corpus(retry_query)
            all_source_groups.append(retry_sources)
            current_sources = self._merge_sources(all_source_groups)
            evidence_check = self.check_evidence_sufficiency(question, current_sources)
            retry_count += 1

        expanded_sources = self.expand_neighbor_context(current_sources)
        used_neighbor_context = len(expanded_sources) > len(current_sources)

        draft_answer = self.generate_answer(question, expanded_sources)
        draft_audit = self.audit_answer(question, draft_answer, expanded_sources)

        if draft_audit.get("revision_needed", False):
            final_answer = self.revise_answer(question, draft_answer, draft_audit, expanded_sources)
            final_audit = self.audit_answer(question, final_answer, expanded_sources)
            revised = True
        else:
            final_answer = draft_answer
            final_audit = draft_audit
            revised = False

        return {
            "question": question,
            "rewritten_query": rewritten_query,
            "retry_count": retry_count,
            "retry_queries": retry_queries,
            "used_neighbor_context": used_neighbor_context,
            "revision_needed": bool(draft_audit.get("revision_needed", False)),
            "revised": revised,
            "evidence_check": evidence_check,
            "draft_answer": draft_answer,
            "draft_audit": draft_audit,
            "final_answer": final_answer,
            "final_audit": final_audit,
            "retrieval_snapshot": {
                "initial_sources": initial_sources,
                "merged_sources_before_expansion": current_sources,
            },
            "final_sources": expanded_sources,
            "trace": {
                "workflow": [
                    "rewrite_query",
                    "search_corpus",
                    "check_evidence_sufficiency",
                    "retry_retrieval_if_needed",
                    "expand_neighbor_context",
                    "generate_answer",
                    "audit_answer",
                    "revise_answer_if_needed",
                ]
            },
        }
