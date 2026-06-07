#src/indexing/inspect_chunks.py
import json
from pathlib import Path
from collections import Counter


def main():
    pages_path = Path("data/parsed/thesis_clean_pages.jsonl")

    if not pages_path.exists():
        raise FileNotFoundError(pages_path)

    print("This script checks raw parsed pages only.")
    print("For final embedded chunks, inspect build_index.py output.")
    print("------------------------------")

    total_pages = 0
    for line in pages_path.open("r", encoding="utf-8"):
        item = json.loads(line)
        total_pages += 1
        page = item.get("page_index")
        parser = item.get("parser")
        text = item.get("text", "")
        print(f"page={page}, parser={parser}, chars={len(text)}")

    print(f"\nTotal pages: {total_pages}")


if __name__ == "__main__":
    main()