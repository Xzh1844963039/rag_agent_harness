#app.py

#app.py

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
QUERY_SCRIPT = PROJECT_ROOT / "src" / "rag" / "query_thesis.py"


def decode_subprocess_output(raw: bytes) -> str:
    """
    Decode subprocess output robustly on Windows.

    Chinese output may be encoded as UTF-8 or GBK depending on the Python/terminal
    environment. This fallback avoids garbled text in the Streamlit UI.
    """
    if not raw:
        return ""

    for encoding in ("utf-8", "gbk", "mbcs"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue

    return raw.decode("utf-8", errors="replace")


def run_query(question: str, show_rewritten_query: bool = True) -> str:
    if not QUERY_SCRIPT.exists():
        raise FileNotFoundError(f"Query script not found: {QUERY_SCRIPT}")

    cmd = [
        sys.executable,
        str(QUERY_SCRIPT),
        question,
    ]

    if show_rewritten_query:
        cmd.append("--show_rewritten_query")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=False,
        env=env,
    )

    stdout = decode_subprocess_output(result.stdout)
    stderr = decode_subprocess_output(result.stderr)

    if result.returncode != 0:
        error_msg = stderr.strip() or stdout.strip()
        raise RuntimeError(error_msg)

    return stdout


def extract_block(text: str, title: str) -> str:
    """
    Robustly extract a block from CLI output.

    Expected CLI format:

    Question
    --------
    ...

    Retrieval Query
    ---------------
    ...

    Answer
    ------
    ...

    Evidence
    --------
    ...
    """
    lines = text.splitlines()
    start = None

    for i in range(len(lines) - 1):
        if lines[i].strip() == title and set(lines[i + 1].strip()) <= {"-"} and lines[i + 1].strip():
            start = i + 2
            break

    if start is None:
        return ""

    end = len(lines)
    for j in range(start, len(lines) - 1):
        current = lines[j].strip()
        next_line = lines[j + 1].strip()
        if current in {"Question", "Retrieval Query", "Answer", "Evidence"} and set(next_line) <= {"-"} and next_line:
            end = j
            break

    return "\n".join(lines[start:end]).strip()


def parse_evidence(raw_evidence: str) -> list[dict[str, str]]:
    if not raw_evidence.strip():
        return []

    chunks = re.split(r"\n(?=\[\d+\]\s+page=)", raw_evidence.strip())
    evidence_items: list[dict[str, str]] = []

    for chunk in chunks:
        lines = [line.rstrip() for line in chunk.splitlines() if line.strip()]
        if not lines:
            continue

        header = lines[0]
        item = {
            "header": header,
            "title": "",
            "preview": "",
        }

        title_lines = []
        preview_lines = []
        active_field = None

        for line in lines[1:]:
            stripped = line.strip()

            if stripped.lower().startswith("title:"):
                active_field = "title"
                title_lines.append(stripped[len("title:"):].strip())
            elif stripped.lower().startswith("preview:"):
                active_field = "preview"
                preview_lines.append(stripped[len("preview:"):].strip())
            else:
                if active_field == "title":
                    title_lines.append(stripped)
                elif active_field == "preview":
                    preview_lines.append(stripped)

        item["title"] = " ".join(title_lines).strip()
        item["preview"] = " ".join(preview_lines).strip()
        evidence_items.append(item)

    return evidence_items


def main() -> None:
    st.set_page_config(
        page_title="Corpus-aware Agentic RAG Demo",
        page_icon="📚",
        layout="wide",
    )

    st.title("📚 Corpus-aware Agentic RAG Demo")

    st.markdown(
        """
This demo runs the local Agentic RAG pipeline over an academic PDF corpus.

It supports query rewriting, grounded answer generation, and evidence inspection.
        """.strip()
    )

    with st.sidebar:
        st.header("Settings")

        show_rewritten_query = st.checkbox(
            "Show rewritten retrieval query",
            value=True,
        )

        st.markdown("---")

        st.subheader("Demo Questions")

        demo_questions = [
            "Does the thesis prove that local CoT repair works for all LLMs and all reasoning benchmarks?",
            "How is the Teacher-Student-Controller framework connected to the final math500 strict improvements?",
            "What is the difference between Table 2 and Table 3 in the thesis?",
            "这篇文章主要在讲什么？",
            "论文有没有证明这个方法可以直接迁移到所有数学数据集？如果没有，应该怎么谨慎表述？",
        ]

        selected_demo = st.radio(
            "Choose a demo question",
            demo_questions,
            index=0,
        )

        if st.button("Use selected question"):
            st.session_state["question"] = selected_demo

    if "question" not in st.session_state:
        st.session_state["question"] = (
            "Does the thesis prove that local CoT repair works for all LLMs and all reasoning benchmarks?"
        )

    question = st.text_area(
        "Question",
        key="question",
        height=100,
        placeholder="Ask a question about the indexed PDF corpus...",
    )

    run_button = st.button("Run Agentic RAG", type="primary")

    if run_button:
        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Running Agentic RAG pipeline..."):
            try:
                output = run_query(
                    question=question.strip(),
                    show_rewritten_query=show_rewritten_query,
                )
            except Exception as exc:
                st.error("Failed to run the query script.")
                st.exception(exc)
                return

        parsed_question = extract_block(output, "Question")
        rewritten_query = extract_block(output, "Retrieval Query")
        answer = extract_block(output, "Answer")
        raw_evidence = extract_block(output, "Evidence")
        evidence_items = parse_evidence(raw_evidence)

        st.markdown("## Question")
        st.write(parsed_question or question)

        if show_rewritten_query and rewritten_query:
            st.markdown("## Retrieval Query")
            st.code(rewritten_query, language="text")

        st.markdown("## Answer")
        if answer:
            st.markdown(answer)
        else:
            st.code(output, language="text")

        st.markdown("## Evidence")

        if not evidence_items:
            st.info("No structured evidence was parsed. Raw output is shown below.")
            st.code(raw_evidence or output, language="text")
        else:
            for idx, item in enumerate(evidence_items, start=1):
                with st.expander(
                    f"Evidence {idx}: {item['header']}",
                    expanded=idx <= 3,
                ):
                    if item["title"]:
                        st.markdown(f"**Title:** {item['title']}")
                    if item["preview"]:
                        st.markdown("**Preview:**")
                        st.write(item["preview"])

        with st.expander("Raw CLI Output"):
            st.code(output, language="text")


if __name__ == "__main__":
    main()