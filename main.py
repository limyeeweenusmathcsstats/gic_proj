import os
import re
from pathlib import Path

from document_processor import DocumentProcessor
from embedder import DocumentEmbedder
from vector_index import VectorIndex
from retriever import Retriever
from groq_llm import GroqLLM
from rag_pipeline import RAGPipeline

TEXT_FILES_DIR = "data/text_files"
VECTOR_STORE_DIR = "data/vector_store"


def choose_pdf() -> str:
    pdfs = sorted(Path(TEXT_FILES_DIR).glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {TEXT_FILES_DIR}")

    print("Available documents:")
    for i, p in enumerate(pdfs, 1):
        print(f"  {i}. {p.name}")

    while True:
        choice = input("Select a document (enter number): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(pdfs):
            return str(pdfs[int(choice) - 1])
        print(f"Please enter a number between 1 and {len(pdfs)}.")


def build_index(pdf_path: str, index: VectorIndex) -> DocumentEmbedder:
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    chunks = processor.split_document(pdf_path)

    embedder = DocumentEmbedder()
    vectors = embedder.embed_documents([chunk.page_content for chunk in chunks])

    index.add_chunks(chunks, vectors)
    return embedder


def log_result(pdf_path: str, question: str, result: dict) -> str:
    log_dir = Path("testing_log")
    log_dir.mkdir(exist_ok=True)

    doc_name = Path(pdf_path).stem
    slug = re.sub(r"[^\w\s]", "", question.lower())
    slug = "_".join(slug.split()[:6])
    log_path = log_dir / f"{doc_name}_{slug}.txt"

    separator = "=" * 60

    lines = [
        separator,
        f"  Document : {Path(pdf_path).name}",
        separator,
        "",
        "QUESTION",
        "--------",
        question,
        "",
        "ANSWER",
        "------",
        result["answer"],
        "",
        f"CONFIDENCE : {result['confidence']}",
        "",
        "SOURCES",
        "-------",
    ]

    for i, s in enumerate(result["sources"], 1):
        lines.append(f"[{i}] Page {s['page']}  |  {s['source_file']}")
        lines.append(f"    Score   : {s['score']}")
        lines.append(f"    Content : {s['content']}")
        lines.append("")

    lines.append(separator)

    log_path.write_text("\n".join(lines), encoding="utf-8")
    return str(log_path)


def main():
    pdf_path = choose_pdf()
    index_name = Path(pdf_path).stem + "_index"

    index = VectorIndex(index_name=index_name, storage_dir=VECTOR_STORE_DIR)

    if index.count == 0:
        embedder = build_index(pdf_path, index)
    else:
        embedder = DocumentEmbedder()

    retriever = Retriever(index, embedder)
    llm = GroqLLM()
    pipeline = RAGPipeline(retriever, llm)

    question = input("Ask a question about the document: ")
    result = pipeline.ask(question)

    log_path = log_result(pdf_path, question, result)

    print("\nAnswer:", result["answer"])
    print("\nSources:")
    for i, s in enumerate(result["sources"], 1):
        print(f"  [{i}] Page {s['page']} (score {s['score']}) — {s['content'][:120]}...")
    print(f"\nConfidence : {result['confidence']}")
    print(f"\nFull output saved to: {log_path}")

if __name__ == "__main__":
    main()
