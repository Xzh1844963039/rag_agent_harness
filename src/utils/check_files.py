#C:\Users\18449\PycharmProjects\rag_agent_harness\src\utils\check_files.py
from pathlib import Path
import yaml


def main():
    config_path = Path("configs/baseline.yaml")

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_pdf = Path(config["paths"]["raw_pdf"])

    print("File check")
    print("----------")
    print(f"Config path: {config_path}")
    print(f"Raw PDF path: {raw_pdf}")
    print(f"Raw PDF exists: {raw_pdf.exists()}")

    if not raw_pdf.exists():
        raise FileNotFoundError(
            f"PDF not found: {raw_pdf}\n"
            "Please copy your thesis PDF into data/raw_docs/ and rename it correctly."
        )

    print(f"PDF size: {raw_pdf.stat().st_size / 1024 / 1024:.2f} MB")
    print("File paths look good.")


if __name__ == "__main__":
    main()
def main():
    config_path = Path("configs/baseline.yaml")

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_pdf = Path(config["paths"]["raw_pdf"])

    print("File check")
    print("----------")
    print(f"Config path: {config_path}")
    print(f"Raw PDF path: {raw_pdf}")
    print(f"Raw PDF exists: {raw_pdf.exists()}")

    if not raw_pdf.exists():
        raise FileNotFoundError(
            f"PDF not found: {raw_pdf}\n"
            "Please copy your thesis PDF into data/raw_docs/ and rename it correctly."
        )

    print(f"PDF size: {raw_pdf.stat().st_size / 1024 / 1024:.2f} MB")
    print("File paths look good.")


if __name__ == "__main__":
    main()