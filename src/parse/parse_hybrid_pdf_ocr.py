#src/parse/parse_hybrid_pdf_ocr.py
import json
import shutil
from pathlib import Path

import fitz  # PyMuPDF
import yaml
from rapidocr_onnxruntime import RapidOCR


def load_config(config_path: str = "configs/baseline.yaml") -> dict:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# 这些页在你的论文里主要包含中文内容，PyMuPDF/LlamaParse 文本层会乱码，
# 所以直接 OCR，不再依赖 PDF 文本层。
FORCE_OCR_PAGES = {2, 3, 4, 5, 6, 32}


MOJIBAKE_MARKERS = [
    "璇", "鎵", "胯", "鏂", "鍦", "鏈", "瀵", "浣", "锛", "紝",
    "銆", "€", "", "", "鏄", "涓", "鐨", "妯", "瀛",
    "鐞", "鎽", "鍏", "骞", "寮", "庣", "綔", "泦", "枃"
]


def mojibake_score(text: str) -> int:
    return sum(text.count(m) for m in MOJIBAKE_MARKERS)


def should_use_ocr(page_idx: int, extracted_text: str) -> bool:
    """
    判断这一页是否应该直接 OCR。
    核心逻辑：
    1. 指定中文页直接 OCR
    2. 如果文本层出现明显中文乱码，也直接 OCR
    """
    if page_idx in FORCE_OCR_PAGES:
        return True

    if not extracted_text.strip():
        return True

    score = mojibake_score(extracted_text)

    # 只要乱码特征比较明显，就不要信任文本层
    if score >= 3:
        return True

    return False


def render_page_to_image(page: fitz.Page, image_path: Path, zoom: float = 3.0) -> None:
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    pix.save(str(image_path))


def run_ocr(engine: RapidOCR, image_path: Path) -> str:
    result = engine(str(image_path))

    # rapidocr 一般返回 (ocr_result, elapse)
    if isinstance(result, tuple):
        result = result[0]

    if not result:
        return ""

    lines = []

    for item in result:
        # item 通常是 [box, text, score]
        try:
            box, text, score = item[0], item[1], item[2]
            x = min(p[0] for p in box)
            y = min(p[1] for p in box)
            lines.append((y, x, text, score))
        except Exception:
            continue

    # 从上到下、从左到右排序
    lines.sort(key=lambda t: (t[0], t[1]))

    cleaned = []
    for _, _, text, score in lines:
        text = str(text).strip()
        if text:
            cleaned.append(text)

    return "\n".join(cleaned)


def main() -> None:
    config = load_config()

    raw_pdf_path = Path(config["paths"]["raw_pdf"])
    output_md_path = Path("data/parsed/thesis_clean.md")
    output_pages_path = Path("data/parsed/thesis_clean_pages.jsonl")
    output_summary_path = Path("data/parsed/thesis_clean_summary.json")
    image_dir = Path("data/parsed/ocr_pages")

    if not raw_pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {raw_pdf_path}")

    output_md_path.parent.mkdir(parents=True, exist_ok=True)

    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    print("Hybrid PDF parser started")
    print("-------------------------")
    print(f"Input PDF: {raw_pdf_path}")
    print(f"Output markdown: {output_md_path}")
    print(f"Force OCR pages: {sorted(FORCE_OCR_PAGES)}")

    pdf = fitz.open(raw_pdf_path)
    ocr_engine = RapidOCR()

    markdown_parts = []
    page_records = []

    ocr_pages = []
    text_pages = []

    for page_idx, page in enumerate(pdf, start=1):
        extracted_text = page.get_text("text")
        extracted_text = extracted_text.replace("\x00", "").replace("￾", "").strip()

        score = mojibake_score(extracted_text)
        use_ocr = should_use_ocr(page_idx, extracted_text)

        if use_ocr:
            image_path = image_dir / f"page_{page_idx:03d}.png"
            render_page_to_image(page, image_path, zoom=3.0)
            final_text = run_ocr(ocr_engine, image_path)
            parser_used = "rapidocr"
            ocr_pages.append(page_idx)
        else:
            final_text = extracted_text
            parser_used = "pymupdf"
            text_pages.append(page_idx)

        markdown_parts.append(
            f"\n\n<!-- page: {page_idx}, parser: {parser_used}, mojibake_score: {score} -->\n\n"
        )
        markdown_parts.append(final_text)

        page_records.append(
            {
                "page_index": page_idx,
                "page_label": str(page_idx),
                "parser": parser_used,
                "mojibake_score": score,
                "text": final_text,
                "metadata": {
                    "source": str(raw_pdf_path),
                    "page": page_idx,
                    "parser": parser_used,
                    "mojibake_score": score,
                },
            }
        )

        preview = final_text.replace("\n", " ")[:80]
        print(
            f"Page {page_idx:02d}: {parser_used}, "
            f"mojibake_score={score}, chars={len(final_text)}, preview={preview}"
        )

    full_text = "\n".join(markdown_parts).strip()
    output_md_path.write_text(full_text, encoding="utf-8")

    with output_pages_path.open("w", encoding="utf-8") as f:
        for item in page_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "source_pdf": str(raw_pdf_path),
        "parsed_markdown": str(output_md_path),
        "pages_jsonl": str(output_pages_path),
        "num_pages": len(page_records),
        "num_characters": len(full_text),
        "ocr_pages": ocr_pages,
        "text_pages": text_pages,
        "force_ocr_pages": sorted(FORCE_OCR_PAGES),
        "parser": "hybrid_pymupdf_rapidocr",
    }

    output_summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nHybrid PDF parser finished")
    print("--------------------------")
    print(f"Total pages: {len(page_records)}")
    print(f"OCR pages: {ocr_pages}")
    print(f"Text pages: {text_pages}")
    print(f"Characters: {len(full_text)}")
    print(f"Saved markdown to: {output_md_path}")
    print(f"Saved page JSONL to: {output_pages_path}")
    print(f"Saved summary to: {output_summary_path}")


if __name__ == "__main__":
    main()