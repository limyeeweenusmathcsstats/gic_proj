from document_processor import DocumentProcessor
from embedder import DocumentEmbedder
from vector_index import VectorIndex
from retriever import Retriever
from groq_llm import GroqLLM
from rag_pipeline import RAGPipeline

PDF_PATH = "data/text_files/google.pdf"
VECTOR_STORE_DIR = "data/vector_store"
INDEX_NAME = "google_index"


def build_index(index):
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    chunks = processor.split_document(PDF_PATH)

    embedder = DocumentEmbedder()
    vectors = embedder.embed_documents([chunk.page_content for chunk in chunks])

    index.add_chunks(chunks, vectors)
    return embedder


def main():
    index = VectorIndex(index_name=INDEX_NAME, storage_dir=VECTOR_STORE_DIR)

    # only ingest if the index is empty
    if index.count == 0:
        embedder = build_index(index)
    
    else:
        embedder = DocumentEmbedder()

    retriever = Retriever(index, embedder)
    llm = GroqLLM()
    pipeline = RAGPipeline(retriever, llm)

    question = input("Ask a question about the document: ")
    result = pipeline.ask(question)

    print("\nAnswer:", result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"page {s['page']} - {s['preview']}")
    print("\nConfidence:", result["confidence"])

if __name__ == "__main__":
    main()
