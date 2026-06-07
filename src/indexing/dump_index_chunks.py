#src/indexing/dump_index_chunks.py
import json
from pathlib import Path

import yaml
from llama_index.core import StorageContext, load_index_from_storage


def load_config(config_path: str = "configs/baseline.yaml") -> dict:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config()
    storage_dir = Path(config["paths"]["storage_dir"])

    if not storage_dir.exists():
        raise FileNotFoundError(
            f"Index storage dir not found: {storage_dir}\n"
            "Please run: python src\\indexing\\build_index.py"
        )

    output_dir = Path("outputs/chunk_debug")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_jsonl = output_dir / "index_chunks.jsonl"
    output_md = output_dir / "index_chunks.md"

    storage_context = StorageContext.from_defaults(
        persist_dir=str(storage_dir)
    )
    index = load_index_from_storage(storage_context)

    # LlamaIndex 的节点实际存在 docstore 里
    docs = index.docstore.docs

    chunks = []

    for node_id, node in docs.items():
        text = node.get_content()
        metadata = node.metadata or {}

        chunk = {
            "node_id": node_id,
            "chunk_id": metadata.get("chunk_id"),
            "page": metadata.get("page"),
            "page_chunk_id": metadata.get("page_chunk_id"),
            "section_type": metadata.get("section_type"),
            "section_title": metadata.get("section_title"),
            "parser": metadata.get("parser"),
            "char_len": metadata.get("char_len", len(text)),
            "chunking": metadata.get("chunking"),
            "text": text,
        }
        chunks.append(chunk)

    # 按 page -> page_chunk_id -> chunk_id 排序，方便人工看
    chunks.sort(
        key=lambda x: (
            x.get("page") if x.get("page") is not None else 9999,
            x.get("page_chunk_id") if x.get("page_chunk_id") is not None else 9999,
            x.get("chunk_id") if x.get("chunk_id") is not None else 9999,
        )
    )

    with output_jsonl.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    with output_md.open("w", encoding="utf-8") as f:
        f.write("# Dumped Index Chunks\n\n")
        f.write(f"Total chunks: {len(chunks)}\n\n")

        for chunk in chunks:
            f.write("---\n\n")
            f.write(
                f"## chunk_id={chunk['chunk_id']} | "
                f"page={chunk['page']} | "
                f"page_chunk_id={chunk['page_chunk_id']} | "
                f"section_type={chunk['section_type']} | "
                f"char_len={chunk['char_len']}\n\n"
            )
            f.write(f"section_title: {chunk['section_title']}\n\n")
            f.write(f"parser: {chunk['parser']}\n\n")
            f.write("```text\n")
            f.write(chunk["text"])
            f.write("\n```\n\n")

    print("Dump index chunks finished")
    print("--------------------------")
    print(f"Storage dir: {storage_dir}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Saved JSONL: {output_jsonl}")
    print(f"Saved Markdown: {output_md}")

    # 顺便打印几个关键页的 chunk 概览
    target_pages = {14, 15, 21, 22, 23}
    print("\nKey page chunk summary")
    print("----------------------")

    for chunk in chunks:
        if chunk.get("page") in target_pages:
            text_preview = chunk["text"].replace("\n", " ")
            text_preview = text_preview[:180]
            print(
                f"page={chunk['page']}, "
                f"page_chunk_id={chunk['page_chunk_id']}, "
                f"chunk_id={chunk['chunk_id']}, "
                f"char_len={chunk['char_len']}, "
                f"section_type={chunk['section_type']}"
            )
            print(f"preview: {text_preview}")
            print()


if __name__ == "__main__":
    main()