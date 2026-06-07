#src/indexing/build_index.py
"""
Generic multi-document index builder for LlamaIndex RAG.

This replaces the older thesis-specific build_index.py.

Main changes:
1. Supports one parsed-pages JSONL file containing one or many documents.
2. Does not hardcode thesis title, doc_type, page ranges, or paper-specific sections.
3. Adds document-level metadata to every chunk:
   doc_id, paper_id, title, authors, year, source, file_name.
4. Automatically infers section_type/section_title from headings and text patterns.
5. Keeps table / figure / algorithm / equation-like blocks safer as complete units.
6. Preserves previous_chunk_id / next_chunk_id globally and doc_previous_chunk_id /
   doc_next_chunk_id within each document.
7. Keeps backward-compatible fields used by your AgenticRAG/eval code:
   page, page_end, section_type, section_title, unit_type, chunk_id,
   page_chunk_id, previous_chunk_id, next_chunk_id, parser, char_len, chunking.

Expected input JSONL format, one page per line. The script is defensive and accepts
several common field names:
{
  "doc_id": "paper_001",                 # optional
  "paper_id": "paper_001",               # optional
  "title": "Paper title",                # optional
  "authors": ["A", "B"],                # optional
  "year": "2025",                       # optional
  "source": "file.pdf",                  # optional
  "file_name": "file.pdf",               # optional
  "page": 1,                             # optional
  "page_index": 0,                       # optional
  "text": "page text...",
  "parser": "pymupdf"                   # optional
}

Run:
python src/indexing/build_index.py

Optional:
python src/indexing/build_index.py --config configs/baseline.yaml
python src/indexing/build_index.py --parsed_pages data/parsed/corpus_pages.jsonl --storage_dir storage/corpus_index
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


# Loaded from configs/corpus_profile.yaml at runtime.
# Keeping these as runtime configuration avoids hardcoding corpus-specific
# titles, benchmark names, or chapter mappings in the Python logic.
CORPUS_DEFAULTS: Dict[str, Any] = {}
CHAPTER_SECTION_MAP: Dict[str, str] = {}


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_corpus_profile(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    profile = load_yaml(path)
    corpus = profile.get("corpus", {}) or {}
    section_rules = corpus.get("section_rules", {}) or {}

    global CORPUS_DEFAULTS, CHAPTER_SECTION_MAP
    CORPUS_DEFAULTS = {
        "corpus_name": corpus.get("name"),
        "title": corpus.get("title"),
        "source": corpus.get("source"),
        "file_name": corpus.get("file_name"),
        "authors": corpus.get("authors", []),
        "year": corpus.get("year", "unknown"),
        "doc_type": corpus.get("doc_type", "document"),
    }
    CHAPTER_SECTION_MAP = {str(k): str(v) for k, v in (section_rules.get("chapter_map", {}) or {}).items()}
    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a generic multi-document LlamaIndex vector index.")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Path to baseline YAML config.")
    parser.add_argument("--parsed_pages", default=None, help="Override parsed pages JSONL path.")
    parser.add_argument("--storage_dir", default=None, help="Override index storage directory.")
    parser.add_argument("--corpus_profile", default="configs/corpus_profile.yaml", help="Optional corpus profile YAML.")
    parser.add_argument("--reset", action="store_true", help="Remove existing storage directory before building.")
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Text cleanup
# -----------------------------------------------------------------------------


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_chunk_text(text: str) -> str:
    text = normalize_whitespace(text)
    # Remove repeated page-number-only lines inside chunks.
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"\d{1,4}", stripped):
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_noise_chunk(text: str) -> bool:
    s = text.strip()
    if not s:
        return True
    if re.fullmatch(r"\d{1,4}", s):
        return True
    if len(s) < 12 and not re.search(r"[A-Za-z\u4e00-\u9fff]", s):
        return True
    # Very symbol-heavy fragments are usually parser noise.
    if len(s) >= 20:
        alnum = len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", s))
        if alnum / max(len(s), 1) < 0.18:
            return True
    return False


# -----------------------------------------------------------------------------
# Metadata helpers
# -----------------------------------------------------------------------------


def stable_id(value: str, prefix: str = "doc") -> str:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:12]
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")[:40]
    if not slug:
        slug = digest
    return f"{prefix}_{slug}_{digest}"


def coerce_page(record: Dict[str, Any]) -> int:
    # Accept both 1-based page and 0-based page_index. Prefer explicit page.
    if record.get("page") is not None:
        try:
            return int(record["page"])
        except Exception:
            pass
    if record.get("page_index") is not None:
        try:
            return int(record["page_index"]) + 1
        except Exception:
            pass
    return 1


def infer_doc_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    source = (
        record.get("source")
        or record.get("file_name")
        or record.get("file")
        or record.get("pdf_path")
        or record.get("path")
        or CORPUS_DEFAULTS.get("source")
        or CORPUS_DEFAULTS.get("file_name")
        or CORPUS_DEFAULTS.get("corpus_name")
        or "unknown_source"
    )
    file_name = record.get("file_name") or CORPUS_DEFAULTS.get("file_name") or Path(str(source)).name

    title = record.get("title") or record.get("doc_title") or record.get("paper_title") or CORPUS_DEFAULTS.get("title")
    if not title:
        # Use file stem as a neutral fallback instead of a hardcoded thesis title.
        title = Path(str(file_name)).stem if file_name else "Untitled document"

    raw_doc_id = record.get("doc_id") or record.get("paper_id") or record.get("document_id")
    if raw_doc_id:
        doc_id = str(raw_doc_id)
    else:
        doc_id = stable_id(f"{source}::{title}", prefix="doc")

    paper_id = str(record.get("paper_id") or doc_id)
    authors = record.get("authors") or record.get("author") or CORPUS_DEFAULTS.get("authors") or []
    if isinstance(authors, str):
        authors_value: Any = authors
    else:
        authors_value = list(authors) if isinstance(authors, list) else []

    year = record.get("year") or record.get("publication_year") or record.get("date") or CORPUS_DEFAULTS.get("year") or "unknown"

    return {
        "doc_id": doc_id,
        "paper_id": paper_id,
        "title": title,
        "authors": authors_value,
        "year": str(year),
        "source": str(source),
        "file_name": str(file_name),
        "doc_type": record.get("doc_type", CORPUS_DEFAULTS.get("doc_type", "document")),
    }


def infer_title_from_early_pages(records: List[Dict[str, Any]]) -> Optional[str]:
    early_text = "\n".join(str(r.get("text") or r.get("content") or "") for r in records[:8])
    lines = [normalize_whitespace(x) for x in early_text.splitlines() if normalize_whitespace(x)]

    for i, line in enumerate(lines):
        if re.fullmatch(r"thesis title\s*:?", line, flags=re.I):
            title_lines: List[str] = []
            for nxt in lines[i + 1 : i + 8]:
                if re.match(r"^(student name|student id|department|program|thesis advisor|date)\s*:?", nxt, flags=re.I):
                    break
                if len(nxt) <= 90:
                    title_lines.append(nxt)
            title = " ".join(title_lines).strip(" :-")
            if title:
                return re.sub(r"\s+", " ", title)

    # Fallback: use corpus profile title if it is specific enough.
    profile_title = CORPUS_DEFAULTS.get("title")
    if profile_title and str(profile_title).lower() not in {"research document corpus", "untitled document"}:
        return str(profile_title)
    return None


def enrich_missing_doc_metadata(records: List[Dict[str, Any]], pages_path: Path) -> List[Dict[str, Any]]:
    inferred_title = infer_title_from_early_pages(records)
    fallback_source = CORPUS_DEFAULTS.get("source") or str(pages_path)
    fallback_file_name = CORPUS_DEFAULTS.get("file_name") or pages_path.name

    enriched: List[Dict[str, Any]] = []
    for record in records:
        r = dict(record)
        r.setdefault("source", fallback_source)
        r.setdefault("file_name", fallback_file_name)
        if inferred_title and not (r.get("title") or r.get("doc_title") or r.get("paper_title")):
            r["title"] = inferred_title
        if CORPUS_DEFAULTS.get("authors") and not r.get("authors"):
            r["authors"] = CORPUS_DEFAULTS.get("authors")
        if CORPUS_DEFAULTS.get("year") and not r.get("year"):
            r["year"] = CORPUS_DEFAULTS.get("year")
        enriched.append(r)
    return enriched


# -----------------------------------------------------------------------------
# Section inference
# -----------------------------------------------------------------------------


SECTION_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("abstract", re.compile(r"^\s*(abstract|摘要)\s*$", re.I)),
    ("introduction", re.compile(r"^\s*((\d+\.?\s*)?introduction|引言|绪论)\s*$", re.I)),
    ("related_work", re.compile(r"^\s*((\d+\.?\s*)?(related work|background|literature review)|相关工作|研究背景)\s*$", re.I)),
    ("method", re.compile(r"^\s*((\d+\.?\s*)?(method|methods|methodology|approach|proposed method|framework)|方法|方法设计|模型|框架)\s*$", re.I)),
    ("setup", re.compile(r"^\s*((\d+\.?\s*)?(experimental setup|experiment setup|implementation details|dataset|datasets|training setup)|实验设置|数据集|实现细节)\s*$", re.I)),
    ("results", re.compile(r"^\s*((\d+\.?\s*)?(results|experiments|evaluation|analysis|discussion)|实验结果|结果|分析|讨论)\s*$", re.I)),
    ("limitation", re.compile(r"^\s*((\d+\.?\s*)?(limitations?|discussion and limitations?)|局限|不足)\s*$", re.I)),
    ("conclusion", re.compile(r"^\s*((\d+\.?\s*)?(conclusion|conclusions|future work|conclusion and future work)|结论|未来工作|总结)\s*$", re.I)),
    ("references", re.compile(r"^\s*(references|bibliography|参考文献)\s*$", re.I)),
    ("acknowledgement", re.compile(r"^\s*(acknowledg(e)?ments?|致谢)\s*$", re.I)),
    ("toc", re.compile(r"^\s*(contents|table of contents|目录)\s*$", re.I)),
]


GENERIC_HEADING_RE = re.compile(
    r"^\s*((chapter\s+\d+|section\s+\d+|\d+(\.\d+){0,3})\s+)?"
    r"[A-Z][A-Za-z0-9 ,:;\-–—/&()]{2,100}\s*$"
)


CHINESE_HEADING_RE = re.compile(r"^\s*(第[一二三四五六七八九十0-9]+[章节部分].{0,60}|[一二三四五六七八九十0-9]+[、.．]\s*.{2,60})\s*$")


def detect_heading(line: str) -> Optional[Tuple[str, str]]:
    cleaned = line.strip().strip("#").strip()
    if not cleaned:
        return None

    for section_type, pattern in SECTION_PATTERNS:
        if pattern.match(cleaned):
            return section_type, cleaned

    # Optional corpus-level chapter mapping, e.g. 1 -> introduction, 2 -> related_work.
    # This is configured in corpus_profile.yaml and can be removed for other corpora.
    m_chapter = re.match(r"^\s*(\d+)(?:\.\d+)*\s+", cleaned)
    if m_chapter and m_chapter.group(1) in CHAPTER_SECTION_MAP:
        return CHAPTER_SECTION_MAP[m_chapter.group(1)], cleaned

    # Generic keyword-based normalization for numbered headings such as
    # "3.1 Overall Framework" or "4.2 Training Setup".
    lower = cleaned.lower()
    keyword_rules = [
        ("abstract", ["abstract"]),
        ("introduction", ["introduction", "motivation", "objective", "organization"]),
        ("related_work", ["related work", "literature", "prior work", "background"]),
        ("method", ["method", "methodology", "approach", "framework", "formulation", "repair", "localization", "controller", "student", "teacher"]),
        ("setup", ["setup", "dataset", "datasets", "training", "implementation"]),
        ("results", ["result", "evaluation", "experiment", "analysis", "difference", "scale"]),
        ("conclusion", ["conclusion", "future work", "limitation"]),
        ("references", ["references", "bibliography"]),
    ]
    if len(cleaned) <= 140:
        for section_type, keywords in keyword_rules:
            if any(k in lower for k in keywords):
                return section_type, cleaned

    if GENERIC_HEADING_RE.match(cleaned) and len(cleaned.split()) <= 14:
        return "section", cleaned

    if CHINESE_HEADING_RE.match(cleaned):
        return "section", cleaned

    return None


def update_section_from_page_text(text: str, current_type: str, current_title: str) -> Tuple[str, str]:
    # Inspect only the beginning of each page. This avoids treating normal body lines as headings.
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    for line in lines[:8]:
        heading = detect_heading(line)
        if heading:
            return heading
    return current_type, current_title


# -----------------------------------------------------------------------------
# Unit extraction
# -----------------------------------------------------------------------------


TABLE_START_RE = re.compile(r"^\s*(table|tab\.|表)\s*\d+", re.I)
FIGURE_START_RE = re.compile(r"^\s*(figure|fig\.|图)\s*\d+", re.I)
ALGORITHM_START_RE = re.compile(r"^\s*(algorithm|算法)\s*\d*", re.I)
EQUATION_HINT_RE = re.compile(r"(\\begin\{equation\}|\\\[|^\s*\(?\d+\)?\s*$)")


def classify_block(block: str) -> str:
    first_line = next((x.strip() for x in block.splitlines() if x.strip()), "")
    if TABLE_START_RE.match(first_line):
        return "table"
    if FIGURE_START_RE.match(first_line):
        return "figure"
    if ALGORITHM_START_RE.match(first_line):
        return "algorithm"
    # Heuristic: pipe-heavy or tabular-looking blocks.
    if block.count("|") >= 4 or re.search(r"\b(row|column|accuracy|score|metric)\b", block, re.I) and len(block.splitlines()) >= 3:
        return "table_like"
    if EQUATION_HINT_RE.search(block) and len(block) < 1200:
        return "equation"
    return "paragraph"


def split_page_blocks(text: str) -> List[str]:
    text = normalize_whitespace(text)
    if not text:
        return []

    # First split by blank lines.
    rough_blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]

    # Further split very long blocks by headings/table/figure starts when the parser did not keep blank lines.
    final_blocks: List[str] = []
    for block in rough_blocks:
        lines = block.splitlines()
        if len(lines) <= 1:
            final_blocks.append(block)
            continue

        buf: List[str] = []
        for line in lines:
            stripped = line.strip()
            starts_new = bool(detect_heading(stripped) or TABLE_START_RE.match(stripped) or FIGURE_START_RE.match(stripped))
            if starts_new and buf:
                final_blocks.append("\n".join(buf).strip())
                buf = [line]
            else:
                buf.append(line)
        if buf:
            final_blocks.append("\n".join(buf).strip())

    return final_blocks


def split_pages_into_units(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []

    # Keep section state per document.
    section_state: Dict[str, Tuple[str, str]] = defaultdict(lambda: ("unknown", "unknown"))

    sorted_records = sorted(
        records,
        key=lambda r: (
            infer_doc_metadata(r)["doc_id"],
            coerce_page(r),
            int(r.get("page_index", coerce_page(r) - 1) or 0),
        ),
    )

    for record in sorted_records:
        text = normalize_whitespace(str(record.get("text") or record.get("content") or ""))
        if not text:
            continue

        doc_meta = infer_doc_metadata(record)
        doc_id = doc_meta["doc_id"]
        page = coerce_page(record)
        page_end = int(record.get("page_end", page) or page)
        parser = record.get("parser", "unknown")

        current_type, current_title = section_state[doc_id]
        current_type, current_title = update_section_from_page_text(text, current_type, current_title)
        section_state[doc_id] = (current_type, current_title)

        for block_idx, block in enumerate(split_page_blocks(text), start=1):
            if is_noise_chunk(block):
                continue

            heading = detect_heading(block.strip()) if len(block.splitlines()) <= 2 else None
            if heading:
                current_type, current_title = heading
                section_state[doc_id] = heading
                # Keep the heading as a small unit only if useful; it can help section retrieval.
                unit_type = "heading"
            else:
                unit_type = classify_block(block)

            unit = {
                **doc_meta,
                "page": page,
                "page_end": page_end,
                "parser": parser,
                "section_type": current_type,
                "section_title": current_title,
                "unit_type": unit_type,
                "block_id": block_idx,
                "text": block,
            }
            units.append(unit)

    return units


# -----------------------------------------------------------------------------
# Chunking
# -----------------------------------------------------------------------------


SENTENCE_RE = re.compile(r"(?<=[。！？.!?])\s+|(?<=[。！？])")


def split_sentences(text: str) -> List[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    parts = [x.strip() for x in SENTENCE_RE.split(text) if x.strip()]
    if len(parts) <= 1:
        # Fallback for long English paragraphs without punctuation spacing.
        parts = [x.strip() for x in re.split(r"(?<=\.) ", text) if x.strip()]
    return parts or [text]


def split_long_unit(unit: Dict[str, Any], max_chars: int, overlap_sentences: int) -> List[Dict[str, Any]]:
    text = clean_chunk_text(unit["text"])
    if len(text) <= max_chars:
        return [{**unit, "text": text}]

    # Never split table/figure/algorithm unless absolutely huge. If huge, split conservatively by lines.
    if unit.get("unit_type") in {"table", "figure", "algorithm", "table_like"}:
        lines = [x for x in text.splitlines() if x.strip()]
        chunks: List[Dict[str, Any]] = []
        buf: List[str] = []
        for line in lines:
            candidate = "\n".join(buf + [line]).strip()
            if buf and len(candidate) > max_chars:
                chunks.append({**unit, "text": "\n".join(buf).strip(), "unit_type": unit.get("unit_type") + "_split"})
                buf = [line]
            else:
                buf.append(line)
        if buf:
            chunks.append({**unit, "text": "\n".join(buf).strip(), "unit_type": unit.get("unit_type") + "_split"})
        return chunks

    sentences = split_sentences(text)
    chunks = []
    buf: List[str] = []

    for sentence in sentences:
        candidate = " ".join(buf + [sentence]).strip()
        if buf and len(candidate) > max_chars:
            chunks.append({**unit, "text": " ".join(buf).strip(), "unit_type": unit.get("unit_type", "paragraph") + "_split"})
            if overlap_sentences > 0:
                buf = buf[-overlap_sentences:] + [sentence]
            else:
                buf = [sentence]
        else:
            buf.append(sentence)

    if buf:
        chunks.append({**unit, "text": " ".join(buf).strip(), "unit_type": unit.get("unit_type", "paragraph") + "_split"})

    return chunks


def can_merge(a: Dict[str, Any], b: Dict[str, Any], max_chars: int) -> bool:
    if a["doc_id"] != b["doc_id"]:
        return False
    if a.get("section_type") != b.get("section_type"):
        return False
    if a.get("unit_type") in {"table", "figure", "algorithm", "table_like"}:
        return False
    if b.get("unit_type") in {"table", "figure", "algorithm", "table_like"}:
        return False
    return len(a.get("text", "")) + len(b.get("text", "")) + 2 <= max_chars


def build_chunks_from_units(
    units: List[Dict[str, Any]],
    max_chars: int,
    min_chunk_chars: int,
    overlap_sentences: int,
) -> List[Dict[str, Any]]:
    # First split overly long units.
    split_units: List[Dict[str, Any]] = []
    for unit in units:
        split_units.extend(split_long_unit(unit, max_chars=max_chars, overlap_sentences=overlap_sentences))

    # Then merge small adjacent paragraph-like units within the same doc/section.
    chunks: List[Dict[str, Any]] = []
    pending: Optional[Dict[str, Any]] = None

    for unit in split_units:
        unit = {**unit, "text": clean_chunk_text(unit.get("text", ""))}
        if is_noise_chunk(unit["text"]):
            continue

        protected = unit.get("unit_type") in {"table", "figure", "algorithm", "table_like", "equation"}

        if pending is None:
            pending = unit
            if protected or len(unit["text"]) >= min_chunk_chars:
                chunks.append(pending)
                pending = None
            continue

        if can_merge(pending, unit, max_chars=max_chars):
            merged_text = (pending["text"].rstrip() + "\n\n" + unit["text"].lstrip()).strip()
            pending = {
                **pending,
                "text": merged_text,
                "page_end": max(int(pending.get("page_end", pending["page"])), int(unit.get("page_end", unit["page"]))),
                "unit_type": "paragraph_merged",
            }
            if len(pending["text"]) >= min_chunk_chars:
                chunks.append(pending)
                pending = None
        else:
            chunks.append(pending)
            pending = unit
            if protected or len(unit["text"]) >= min_chunk_chars:
                chunks.append(pending)
                pending = None

    if pending is not None:
        chunks.append(pending)

    # Final cleanup: avoid tiny heading-only chunks by merging forward when possible.
    cleaned: List[Dict[str, Any]] = []
    i = 0
    while i < len(chunks):
        cur = chunks[i]
        if (
            len(cur.get("text", "")) < min_chunk_chars
            and cur.get("unit_type") == "heading"
            and i + 1 < len(chunks)
            and can_merge(cur, chunks[i + 1], max_chars=max_chars)
        ):
            nxt = chunks[i + 1]
            merged = {
                **cur,
                "text": (cur["text"].rstrip() + "\n\n" + nxt["text"].lstrip()).strip(),
                "page_end": max(int(cur.get("page_end", cur["page"])), int(nxt.get("page_end", nxt["page"]))),
                "unit_type": nxt.get("unit_type", "paragraph"),
            }
            cleaned.append(merged)
            i += 2
        else:
            cleaned.append(cur)
            i += 1

    return cleaned


# -----------------------------------------------------------------------------
# Node creation and metadata assignment
# -----------------------------------------------------------------------------


def assign_chunk_metadata(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    page_counter: Dict[Tuple[str, int], int] = defaultdict(int)
    doc_chunks: Dict[str, List[int]] = defaultdict(list)

    # Global chunk id.
    for idx, chunk in enumerate(chunks, start=1):
        doc_id = str(chunk["doc_id"])
        page = int(chunk["page"])
        page_counter[(doc_id, page)] += 1

        chunk["chunk_id"] = idx
        chunk["page_chunk_id"] = page_counter[(doc_id, page)]
        chunk["previous_chunk_id"] = idx - 1 if idx > 1 else None
        chunk["next_chunk_id"] = idx + 1 if idx < len(chunks) else None
        doc_chunks[doc_id].append(idx)

    # Per-document neighbor ids.
    id_to_chunk = {int(c["chunk_id"]): c for c in chunks}
    for doc_id, ids in doc_chunks.items():
        for pos, chunk_id in enumerate(ids):
            chunk = id_to_chunk[chunk_id]
            chunk["doc_previous_chunk_id"] = ids[pos - 1] if pos > 0 else None
            chunk["doc_next_chunk_id"] = ids[pos + 1] if pos + 1 < len(ids) else None

    return chunks


def load_page_records(pages_path: Path) -> List[Dict[str, Any]]:
    if not pages_path.exists():
        raise FileNotFoundError(f"Parsed pages file not found: {pages_path}")

    records: List[Dict[str, Any]] = []
    with pages_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {pages_path}:{line_no}: {e}") from e
            records.append(record)

    if not records:
        raise RuntimeError(f"No page records loaded from: {pages_path}")
    return records


def create_nodes_from_pages(
    pages_path: Path,
    max_chars: int,
    min_chunk_chars: int,
    overlap_sentences: int,
) -> List[TextNode]:
    records = load_page_records(pages_path)
    records = enrich_missing_doc_metadata(records, pages_path=pages_path)
    units = split_pages_into_units(records)

    # Sort by doc, page, block id to keep neighbors meaningful.
    units.sort(key=lambda x: (str(x["doc_id"]), int(x["page"]), int(x.get("block_id", 0))))

    chunks = build_chunks_from_units(
        units=units,
        max_chars=max_chars,
        min_chunk_chars=min_chunk_chars,
        overlap_sentences=overlap_sentences,
    )
    chunks = assign_chunk_metadata(chunks)

    nodes: List[TextNode] = []
    for chunk in chunks:
        text = clean_chunk_text(chunk["text"])
        if is_noise_chunk(text):
            continue

        page = int(chunk["page"])
        page_end = int(chunk.get("page_end", page))
        title = chunk.get("title") or "Untitled document"
        doc_id = str(chunk.get("doc_id"))

        metadata = {
            "doc_id": doc_id,
            "paper_id": str(chunk.get("paper_id", doc_id)),
            "title": title,
            "authors": chunk.get("authors", []),
            "year": str(chunk.get("year", "unknown")),
            "source": chunk.get("source", "unknown_source"),
            "file_name": chunk.get("file_name", "unknown_file"),
            "doc_type": chunk.get("doc_type", "document"),
            "page": page,
            "page_end": page_end,
            "parser": chunk.get("parser", "unknown"),
            "section_type": chunk.get("section_type", "unknown"),
            "section_title": chunk.get("section_title", f"page_{page}"),
            "unit_type": chunk.get("unit_type", "paragraph"),
            "chunk_id": int(chunk["chunk_id"]),
            "page_chunk_id": int(chunk["page_chunk_id"]),
            "previous_chunk_id": chunk.get("previous_chunk_id"),
            "next_chunk_id": chunk.get("next_chunk_id"),
            "doc_previous_chunk_id": chunk.get("doc_previous_chunk_id"),
            "doc_next_chunk_id": chunk.get("doc_next_chunk_id"),
            "chunking": "generic_v1_multi_doc_section_paragraph_table_figure_sentence_safe",
            "char_len": len(text),
        }

        nodes.append(TextNode(text=text, metadata=metadata))

    if not nodes:
        raise RuntimeError("No valid nodes created from parsed pages.")
    return nodes


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def print_build_report(nodes: List[TextNode], max_chars: int) -> None:
    doc_counts = Counter(str(n.metadata.get("doc_id", "unknown")) for n in nodes)
    section_counts = Counter(str(n.metadata.get("section_type", "unknown")) for n in nodes)
    unit_counts = Counter(str(n.metadata.get("unit_type", "unknown")) for n in nodes)
    lengths = [int(n.metadata.get("char_len", len(n.text))) for n in nodes]

    print(f"\nLoaded nodes: {len(nodes)}")

    print("\nDocument counts:")
    for k, v in doc_counts.most_common():
        title = next((n.metadata.get("title") for n in nodes if str(n.metadata.get("doc_id")) == k), "")
        print(f"- {k}: {v} chunks | title={title}")

    print("\nSection counts:")
    for k, v in sorted(section_counts.items()):
        print(f"- {k}: {v}")

    print("\nUnit counts:")
    for k, v in sorted(unit_counts.items()):
        print(f"- {k}: {v}")

    print("\nChunk length stats:")
    print(f"- min: {min(lengths)}")
    print(f"- max: {max(lengths)}")
    print(f"- avg: {sum(lengths) / len(lengths):.1f}")
    print(f"- chunks over max_chars: {sum(1 for x in lengths if x > max_chars)}")
    print(f"- pure page-number chunks: {sum(1 for node in nodes if re.fullmatch(r'\\d{1,4}', node.text.strip()))}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    load_dotenv()
    args = parse_args()

    config = load_yaml(Path(args.config))
    load_corpus_profile(Path(args.corpus_profile) if args.corpus_profile else None)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is missing. Please check your .env file.")

    paths = config.get("paths", {})
    rag_config = config.get("rag", {})
    llm_config = config.get("llm", {})
    embedding_config = config.get("embedding", {})

    parsed_pages_path = Path(args.parsed_pages or paths.get("parsed_pages", "data/parsed/corpus_pages.jsonl"))
    storage_dir = Path(args.storage_dir or paths.get("storage_dir", "storage/corpus_index"))

    llm_model = llm_config.get("model", "gpt-4o-mini")
    embedding_model = embedding_config.get("model", "text-embedding-3-small")

    max_chars = int(rag_config.get("chunk_max_chars", 1800))
    min_chunk_chars = int(rag_config.get("min_chunk_chars", 180))
    overlap_sentences = int(rag_config.get("chunk_overlap_sentences", 1))

    print("Build generic multi-document index started")
    print("------------------------------------------")
    print(f"Parsed pages: {parsed_pages_path}")
    print(f"Storage dir: {storage_dir}")
    print(f"LLM model: {llm_model}")
    print(f"Embedding model: {embedding_model}")
    print(f"Chunk max chars: {max_chars}")
    print(f"Min chunk chars: {min_chunk_chars}")
    print(f"Overlap sentences for long splits only: {overlap_sentences}")
    print("Chunking strategy: generic multi-doc + auto section + paragraph-first + table/figure-safe + sentence-safe split")

    Settings.llm = OpenAI(model=llm_model, api_key=openai_api_key)
    Settings.embed_model = OpenAIEmbedding(model=embedding_model, api_key=openai_api_key)

    nodes = create_nodes_from_pages(
        pages_path=parsed_pages_path,
        max_chars=max_chars,
        min_chunk_chars=min_chunk_chars,
        overlap_sentences=overlap_sentences,
    )

    print_build_report(nodes, max_chars=max_chars)

    if storage_dir.exists():
        if args.reset:
            print(f"\nRemoving existing storage dir: {storage_dir}")
            shutil.rmtree(storage_dir)
        else:
            raise FileExistsError(f"Storage dir already exists: {storage_dir}. Use --reset to rebuild.")

    storage_dir.mkdir(parents=True, exist_ok=True)

    index = VectorStoreIndex(nodes=nodes, show_progress=True)
    index.storage_context.persist(persist_dir=str(storage_dir))

    print("\nBuild generic multi-document index finished")
    print("-------------------------------------------")
    print(f"Index saved to: {storage_dir}")


if __name__ == "__main__":
    main()

