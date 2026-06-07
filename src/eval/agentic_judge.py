#src/eval/agentic_judge.py
"""
Generic Agentic Judge for source-grounded RAG evaluation.

Default mode is online-style judging:
- use_gold_hints=False
- The judge sees only question, answer, and retrieved sources.

Offline benchmark mode is optional:
- use_gold_hints=True
- The judge may also see category / expected behavior / answer keywords.
- This should be used only for benchmark analysis, not for simulating real users.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


class AgenticJudge:
    def __init__(self, llm: Any, use_gold_hints: bool = False) -> None:
        self.llm = llm
        self.use_gold_hints = use_gold_hints

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
        return json.loads(match.group(0))

    @staticmethod
    def _clamp(value: Any, default: float = 0.0) -> float:
        try:
            v = float(value)
        except Exception:
            return default
        return max(0.0, min(5.0, v))

    @staticmethod
    def _format_sources(sources: List[Dict[str, Any]], max_sources: int = 8, max_chars_each: int = 1200) -> str:
        parts = []
        for source in sources[:max_sources]:
            metadata = source.get("metadata", {})
            text = str(source.get("text", "")).replace("\n", " ").strip()
            if len(text) > max_chars_each:
                text = text[:max_chars_each] + "..."
            parts.append(
                f"[Source {source.get('rank')}] page={metadata.get('page')}, "
                f"section={metadata.get('section_type')}, title={metadata.get('section_title')}, "
                f"chunk_id={metadata.get('chunk_id')}\n{text}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _negative_answer(text: str) -> bool:
        markers = [
            "not supported", "does not support", "do not support", "not report", "does not report",
            "not compare", "does not compare", "not prove", "does not prove", "no evidence",
            "cannot conclude", "insufficient evidence", "unsupported",
            "不支持", "没有证据", "未报告", "没有报告", "未比较", "没有比较", "不能证明", "无法证明",
        ]
        lower = text.lower()
        return any(m in lower for m in markers)

    def classify_question(self, item: Dict[str, Any]) -> Dict[str, Any]:
        question = item.get("question", "")
        gold_hint_text = ""
        if self.use_gold_hints:
            gold_hint_text = json.dumps(
                {
                    "category": item.get("category"),
                    "expected_behavior": item.get("expected_behavior"),
                },
                ensure_ascii=False,
            )

        prompt = f"""
Classify this user question for RAG evaluation.

Question:
{question}

Optional offline benchmark hints, only for classification if present:
{gold_hint_text}

Return valid JSON only:
{{
  "question_type": "direct_qa" | "claim_verification" | "comparison" | "table_or_figure" | "limitation" | "cross_section" | "unknown",
  "requires_exact_evidence": true or false,
  "requires_cautious_answer": true or false,
  "reason": "brief reason"
}}
""".strip()

        raw = str(self.llm.complete(prompt)).strip()
        try:
            parsed = self._extract_json_object(raw)
        except Exception:
            parsed = {
                "question_type": "unknown",
                "requires_exact_evidence": True,
                "requires_cautious_answer": False,
                "reason": "Failed to parse classification JSON.",
                "raw_response": raw,
            }

        parsed.setdefault("question_type", "unknown")
        parsed["requires_exact_evidence"] = bool(parsed.get("requires_exact_evidence", True))
        parsed["requires_cautious_answer"] = bool(parsed.get("requires_cautious_answer", False))
        parsed.setdefault("reason", "")
        return parsed

    def extract_and_verify_claims(
        self,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        source_text = self._format_sources(sources)
        prompt = f"""
Extract the important factual claims from the answer and verify each claim against the retrieved sources.

Question:
{question}

Answer:
{answer}

Retrieved sources:
{source_text}

Verification rules:
1. Each important factual claim must be directly supported by the retrieved sources.
2. Evidence about one benchmark, dataset, model, method, metric, table, figure, or setting does not support a different one unless the sources explicitly connect them.
3. Related work is not the same as an experimental comparison.
4. Universal claims require explicit broad evidence.
5. If the answer correctly says a claim is not supported, mark that cautious refusal as supported when the sources indeed lack direct support.

Return valid JSON only:
{{
  "claims": [
    {{
      "claim": "claim text",
      "status": "supported" | "unsupported" | "partially_supported" | "cautious_refusal",
      "evidence_sources": [1, 2],
      "explanation": "brief explanation"
    }}
  ],
  "summary": "brief summary"
}}
""".strip()

        raw = str(self.llm.complete(prompt)).strip()
        try:
            parsed = self._extract_json_object(raw)
        except Exception:
            parsed = {
                "claims": [],
                "summary": "Failed to parse claim verification JSON.",
                "raw_response": raw,
            }
        if not isinstance(parsed.get("claims"), list):
            parsed["claims"] = []
        parsed.setdefault("summary", "")
        return parsed

    @staticmethod
    def derive_claim_metrics(claim_report: Dict[str, Any]) -> Dict[str, Any]:
        claims = claim_report.get("claims", []) or []
        if not claims:
            return {
                "num_claims": 0,
                "num_supported": 0,
                "num_partial": 0,
                "num_unsupported": 0,
                "num_cautious_refusal": 0,
                "claim_support_rate": 0.0,
                "critical_mismatch_count": 0,
            }

        counts = {
            "supported": 0,
            "partially_supported": 0,
            "unsupported": 0,
            "cautious_refusal": 0,
        }
        critical_mismatch_count = 0
        for claim in claims:
            status = str(claim.get("status", "unsupported"))
            if status in counts:
                counts[status] += 1
            else:
                counts["unsupported"] += 1
            explanation = str(claim.get("explanation", "")).lower()
            # Count a critical mismatch only when the claim itself is unsupported
            # or partially supported. Do NOT penalize supported cautious refusals
            # just because their explanations contain words like "universal" or
            # "unsupported". Otherwise correct refusal answers get falsely
            # downgraded, as seen in hard-eval limitation questions.
            if status in {"unsupported", "partially_supported"} and any(
                x in explanation for x in ["different", "mismatch", "not the same", "universal", "unsupported"]
            ):
                critical_mismatch_count += 1

        support_like = counts["supported"] + counts["cautious_refusal"] + 0.5 * counts["partially_supported"]
        return {
            "num_claims": len(claims),
            "num_supported": counts["supported"],
            "num_partial": counts["partially_supported"],
            "num_unsupported": counts["unsupported"],
            "num_cautious_refusal": counts["cautious_refusal"],
            "claim_support_rate": support_like / len(claims),
            "critical_mismatch_count": critical_mismatch_count,
        }

    def score_with_rubric(
        self,
        item: Dict[str, Any],
        answer: str,
        sources: List[Dict[str, Any]],
        classification: Dict[str, Any],
        claim_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        question = item.get("question", "")
        source_text = self._format_sources(sources)

        gold_hint_text = ""
        if self.use_gold_hints:
            gold_hint_text = json.dumps(
                {
                    "answer_keywords": item.get("answer_keywords", []),
                    "expected_behavior": item.get("expected_behavior"),
                    "expected_section_type": item.get("expected_section_type"),
                },
                ensure_ascii=False,
            )

        prompt = f"""
Score this RAG answer using a strict source-grounded rubric.

Question:
{question}

Answer:
{answer}

Question classification:
{json.dumps(classification, ensure_ascii=False)}

Claim verification report:
{json.dumps(claim_report, ensure_ascii=False)}

Retrieved sources:
{source_text}

Optional offline benchmark hints. Use only when present and only for benchmark analysis:
{gold_hint_text}

Scoring:
- correctness: answers the question correctly.
- completeness: covers necessary points.
- groundedness: claims are grounded in sources.
- citation_support: sources directly support answer.
- hallucination_risk: unsupported or overgeneralized claims.
- overall_score: holistic score.

Return valid JSON only:
{{
  "correctness": 0-5,
  "completeness": 0-5,
  "groundedness": 0-5,
  "citation_support": 0-5,
  "hallucination_risk": 0-5,
  "overall_score": 0-5,
  "verdict": "excellent" | "good" | "partial" | "poor",
  "reason": "brief reason",
  "missing_points": ["missing point 1"],
  "unsupported_points": ["unsupported point 1"]
}}
""".strip()

        raw = str(self.llm.complete(prompt)).strip()
        try:
            parsed = self._extract_json_object(raw)
        except Exception:
            parsed = {
                "correctness": 0,
                "completeness": 0,
                "groundedness": 0,
                "citation_support": 0,
                "hallucination_risk": 5,
                "overall_score": 0,
                "verdict": "poor",
                "reason": "Failed to parse rubric score JSON.",
                "missing_points": [],
                "unsupported_points": [],
                "raw_response": raw,
            }

        for key in ["correctness", "completeness", "groundedness", "citation_support", "hallucination_risk", "overall_score"]:
            parsed[key] = self._clamp(parsed.get(key), default=5.0 if key == "hallucination_risk" else 0.0)
        for key in ["missing_points", "unsupported_points"]:
            if not isinstance(parsed.get(key), list):
                parsed[key] = []
        parsed.setdefault("verdict", "partial")
        parsed.setdefault("reason", "")
        return parsed

    def adjust_scores(
        self,
        item: Dict[str, Any],
        answer: str,
        classification: Dict[str, Any],
        claim_metrics: Dict[str, Any],
        rubric_score: Dict[str, Any],
    ) -> Dict[str, Any]:
        adjusted = dict(rubric_score)

        support_rate = float(claim_metrics.get("claim_support_rate", 0.0))
        unsupported = int(claim_metrics.get("num_unsupported", 0))
        critical = int(claim_metrics.get("critical_mismatch_count", 0))

        if unsupported > 0:
            adjusted["citation_support"] = min(float(adjusted.get("citation_support", 0)), 3.0)
            adjusted["groundedness"] = min(float(adjusted.get("groundedness", 0)), 3.0)
            adjusted["hallucination_risk"] = max(float(adjusted.get("hallucination_risk", 0)), 2.0)

        if critical > 0:
            adjusted["citation_support"] = min(float(adjusted.get("citation_support", 0)), 2.0)
            adjusted["groundedness"] = min(float(adjusted.get("groundedness", 0)), 2.0)
            adjusted["hallucination_risk"] = max(float(adjusted.get("hallucination_risk", 0)), 3.0)
            adjusted["overall_score"] = min(float(adjusted.get("overall_score", 0)), 2.0)

        if support_rate >= 0.95 and adjusted.get("hallucination_risk", 5) <= 1:
            adjusted["overall_score"] = max(float(adjusted.get("overall_score", 0)), 4.0)

        # Claim-verification: a clear cautious refusal is good when the answer says unsupported.
        if classification.get("question_type") == "claim_verification" and self._negative_answer(answer):
            if unsupported == 0:
                adjusted["correctness"] = max(float(adjusted.get("correctness", 0)), 4.0)
                adjusted["groundedness"] = max(float(adjusted.get("groundedness", 0)), 4.0)

        overall = float(adjusted.get("overall_score", 0))
        if overall >= 4.5:
            adjusted["verdict"] = "excellent"
        elif overall >= 3.5:
            adjusted["verdict"] = "good"
        elif overall >= 2.0:
            adjusted["verdict"] = "partial"
        else:
            adjusted["verdict"] = "poor"

        return adjusted

    def evaluate(
        self,
        item: Dict[str, Any],
        answer: str,
        sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        classification = self.classify_question(item)
        claim_report = self.extract_and_verify_claims(item.get("question", ""), answer, sources)
        claim_metrics = self.derive_claim_metrics(claim_report)
        rubric_score = self.score_with_rubric(item, answer, sources, classification, claim_report)
        final_score = self.adjust_scores(item, answer, classification, claim_metrics, rubric_score)

        return {
            "use_gold_hints": self.use_gold_hints,
            "classification": classification,
            "claim_report": claim_report,
            "claim_metrics": claim_metrics,
            "rubric_score": rubric_score,
            "final_score": final_score,
        }
