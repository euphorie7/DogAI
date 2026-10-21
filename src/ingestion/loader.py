from pathlib import Path
 


def load_documents(file_path :str) -> list[str]

    text = Path(file_path).read_text(encoding="utf-8")

    documents = text.split("---DOCUMENT---")

    return [doc.strip() for doc in documents if doc.strip()]
