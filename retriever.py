from typing import List, Dict, Any

from embedder import DocumentEmbedder
from vector_index import VectorIndex


class Retriever:
    """Finds relevant document chunks for a query using vector similarity."""

    def __init__(self, index: VectorIndex, embedder: DocumentEmbedder):
        self._index = index
        self._embedder = embedder

    def retrieve(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """Embed the query and return the top_k most similar chunks above min_score."""
        query_vector = self._embedder.embed_query(query)
        hits = self._index.search(query_vector, top_k=top_k)

        return [
            {
                "content": hit["content"],
                "score": hit["score"],
                "page": hit["metadata"].get("page", "unknown") + 1,
                "source_file": hit["metadata"].get("source_file", "unknown"),
            }
            for hit in hits
            if hit["score"] >= min_score
        ]

import unittest
import tempfile
from langchain_core.documents import Document


class TestRetriever(unittest.TestCase):

    def setUp(self):
        self.embedder = DocumentEmbedder()
        self.index = VectorIndex(index_name="test_retriever_v2", storage_dir=tempfile.mkdtemp())

        chunks = [
            Document(page_content="Google reported strong revenue growth in 2024.", metadata={"source_file": "google.pdf", "page": 1}),
            Document(page_content="The number of full-time employees increased to 180,000.", metadata={"source_file": "google.pdf", "page": 5}),
        ]
        vectors = self.embedder.embed_documents([c.page_content for c in chunks])
        self.index.add_chunks(chunks, vectors)
        self.retriever = Retriever(self.index, self.embedder)

    def test_retrieve_returns_results(self):
        results = self.retriever.retrieve("What was the revenue?", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("content", results[0])

    def test_relevant_chunk_ranked_first(self):
        results = self.retriever.retrieve("annual revenue", top_k=2)
        self.assertIn("revenue", results[0]["content"].lower())


if __name__ == "__main__":
    unittest.main()