from pathlib import Path
from typing import List

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentProcessor:
    """Loads and splits a single PDF into chunks for RAG."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def split_document(self, pdf_path: str) -> List[Document]:
        """
        Load a single PDF and split it into chunks.
        Keeps page number and source file in metadata for evidence grounding.
        """
        path = Path(pdf_path).resolve()

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {path.suffix}")
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        loader = PDFPlumberLoader(str(pdf_path))
        pages = loader.load()

        for page_doc in pages:
            page_doc.metadata["source_file"] = path.name
            page_doc.metadata["source"] = str(path)

        chunks = self._splitter.split_documents(pages)

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

        return chunks


import unittest

SAMPLE_PDF = str(Path(__file__).parent / "data" / "text_files" / "google.pdf")


class TestDocumentProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = DocumentProcessor()

    def test_splits_into_chunks_with_metadata(self):
        chunks = self.processor.split_document(SAMPLE_PDF)
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIn("source_file", chunk.metadata)
            self.assertIn("page", chunk.metadata)
            self.assertIn("chunk_index", chunk.metadata)

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.processor.split_document("data/text_files/nonexistent.pdf")

    def test_non_pdf_rejected(self):
        txt_path = str(Path(__file__).parent / "data" / "text_files" / "test_text_file.txt")
        with self.assertRaises(ValueError):
            self.processor.split_document(txt_path)


if __name__ == "__main__":
    unittest.main()
