#C:\Users\18449\PycharmProjects\rag_agent_harness\src\utils\check_env.py
import os
from dotenv import load_dotenv


def main():
    load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY")
    llama_key = os.getenv("LLAMA_CLOUD_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    print("Environment check")
    print("-----------------")
    print(f"OPENAI_API_KEY exists: {openai_key is not None and len(openai_key) > 0}")
    print(f"LLAMA_CLOUD_API_KEY exists: {llama_key is not None and len(llama_key) > 0}")
    print(f"OPENAI_MODEL: {openai_model}")
    print(f"EMBEDDING_MODEL: {embedding_model}")

    if not openai_key:
        raise ValueError("OPENAI_API_KEY is missing. Please check your .env file.")

    if not llama_key:
        raise ValueError("LLAMA_CLOUD_API_KEY is missing. Please check your .env file.")

    print("Environment looks good.")


if __name__ == "__main__":
    main()