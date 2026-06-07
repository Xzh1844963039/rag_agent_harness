#src/parse/check_clean_quality.py
from pathlib import Path
import json
import re


MD_PATH = Path("data/parsed/thesis_clean.md")
SUMMARY_PATH = Path("data/parsed/thesis_clean_summary.json")

BAD_MARKERS = [
    "璇氫俊",
    "鎽樿",
    "鎵胯",
    "鏈汉",
    "锛",
    "紝",
    "銆",
    "",
    "",
]

# 单个关键词：这些词必须能在解析后的文本中找到
REQUIRED_TERMS = [
    "诚信承诺",
    "摘要",
    "致谢",
    "肖泽昊",
    "Teacher",
    "Student",
    "Controller",
    "Qwen2.5-Math",
    "math500",
]

# 组合关键词：不再要求完整标题一模一样出现，
# 只要求这些核心片段都出现，避免 OCR 换行、断词、空格导致误报
REQUIRED_TERM_GROUPS = {
    "thesis_title": [
        "Student-Oriented",
        "Chain-of-Thought",
        "Optimization",
    ],
}


def normalize_for_check(text: str) -> str:
    """
    用于质量检查的轻量归一化。
    不改变原始文件，只是为了降低 OCR 换行、多个空格、特殊连字符带来的误报。
    """
    text = text.replace("\x00", "")
    text = text.replace("￾", "")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # 将不同类型的连字符归一化
    text = text.replace("‐", "-")
    text = text.replace("-", "-")
    text = text.replace("–", "-")
    text = text.replace("—", "-")

    # 合并多余空白
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def contains_fuzzy(text: str, term: str) -> bool:
    """
    普通包含检查 + 轻量 fuzzy 检查。
    对英文短语，允许中间出现多个空格或换行。
    """
    if term in text:
        return True

    normalized_text = normalize_for_check(text)
    normalized_term = normalize_for_check(term)

    if normalized_term in normalized_text:
        return True

    # 对英文短语做空格宽松匹配
    if re.search(r"[A-Za-z]", term):
        parts = re.split(r"\s+", normalized_term)
        pattern = r"\s+".join(re.escape(p) for p in parts if p)
        return re.search(pattern, normalized_text, flags=re.IGNORECASE) is not None

    return False


def main():
    if not MD_PATH.exists():
        raise FileNotFoundError(f"Parsed markdown not found: {MD_PATH}")

    text = MD_PATH.read_text(encoding="utf-8")
    normalized_text = normalize_for_check(text)

    print("Clean quality check")
    print("-------------------")
    print(f"File: {MD_PATH}")
    print(f"Characters: {len(text)}")

    print("\nRequired terms:")
    missing_terms = []

    for term in REQUIRED_TERMS:
        ok = contains_fuzzy(normalized_text, term)
        print(f"- {term}: {ok}")
        if not ok:
            missing_terms.append(term)

    print("\nRequired term groups:")
    missing_groups = []

    for group_name, terms in REQUIRED_TERM_GROUPS.items():
        group_results = {}
        for term in terms:
            group_results[term] = contains_fuzzy(normalized_text, term)

        ok = all(group_results.values())
        print(f"- {group_name}: {ok}")

        for term, term_ok in group_results.items():
            print(f"  - {term}: {term_ok}")

        if not ok:
            missing_groups.append(group_name)

    print("\nBad marker counts:")
    total_bad = 0

    for marker in BAD_MARKERS:
        count = text.count(marker)
        total_bad += count
        print(f"- {marker}: {count}")

    if SUMMARY_PATH.exists():
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        print("\nParser summary:")
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    if total_bad > 0:
        raise ValueError(
            f"Still found {total_bad} mojibake markers. "
            "Open the file in VS Code or inspect with Python slices to locate them."
        )

    if missing_terms:
        raise ValueError(
            "Missing required terms: "
            + ", ".join(missing_terms)
        )

    if missing_groups:
        raise ValueError(
            "Missing required term groups: "
            + ", ".join(missing_groups)
        )

    print("\nClean text quality looks good.")


if __name__ == "__main__":
    main()