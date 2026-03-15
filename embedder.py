import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer


class DocumentEmbedder:
    """Generates embeddings for document chunks and queries using SentenceTransformer."""

    def __init__(self, encoder_name: str = "all-MiniLM-L6-v2"):
        self.encoder_name = encoder_name
        self._encoder = None
        self._load_encoder()

    def _load_encoder(self):
        self._encoder = SentenceTransformer(self.encoder_name)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        vectors = self._encoder.encode(texts, show_progress_bar=True)
        return vectors

    def embed_query(self, query: str) -> np.ndarray:
        return self._encoder.encode([query])[0]

import unittest


class TestDocumentEmbedder(unittest.TestCase):

    def setUp(self):
        self.embedder = DocumentEmbedder()

    def test_embed_documents_returns_correct_shape(self):
        texts = ["This is a test sentence.", "Another sentence here."]
        vectors = self.embedder.embed_documents(texts)
        self.assertEqual(vectors.shape[0], len(texts))
        self.assertGreater(vectors.shape[1], 0)

    def test_embed_query_returns_1d_vector(self):
        vector = self.embedder.embed_query("What is the revenue?")
        self.assertEqual(vector.ndim, 1)
        self.assertGreater(len(vector), 0)

    def test_similar_texts_have_higher_score_than_unrelated(self):
        v1 = self.embedder.embed_query("What is the annual revenue?")
        v2 = self.embedder.embed_query("What is the total income for the year?")
        v3 = self.embedder.embed_query("How many employees does the company have?")

        # Cosine similarity
        def cosine(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        self.assertGreater(cosine(v1, v2), cosine(v1, v3))


if __name__ == "__main__":
    unittest.main()
