import os
import uuid
import numpy as np
from typing import List, Any

import chromadb
from langchain_core.documents import Document


class VectorIndex:
    """Stores and retrieves document chunk embeddings using ChromaDB."""

    def __init__(self, index_name: str = "document_index", storage_dir: str = "data/vector_store"):
        self.index_name = index_name
        self.storage_dir = storage_dir
        self._client = None
        self._index = None
        self._setup()

    def _setup(self):
        os.makedirs(self.storage_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.storage_dir)
        self._index = self._client.get_or_create_collection(name=self.index_name)
        print(f"Vector index ready: '{self.index_name}' ({self._index.count()} chunks stored)")

    def add_chunks(self, chunks: List[Document], vectors: np.ndarray):
        """
        Store document chunks and their embeddings.
        Each chunk's metadata (page number, source file) is preserved for evidence grounding.
        """
        if len(chunks) != len(vectors):
            raise ValueError("Number of chunks must match number of vectors")

        ids, texts, metadatas, embeddings = [], [], [], []

        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            ids.append(f"chunk_{uuid.uuid4().hex[:8]}_{i}")
            texts.append(chunk.page_content)
            meta = dict(chunk.metadata)
            meta["chunk_index"] = i
            metadatas.append(meta)
            embeddings.append(vector.tolist())

        self._index.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
        print(f"Stored {len(chunks)} chunks. Total in index: {self._index.count()}")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[dict]:
        """
        Find the most relevant chunks for a query vector.
        Returns a list of results with content, metadata, and similarity score.
        """
        results = self._index.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k
        )

        hits = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist, doc_id in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
                results["ids"][0],
            ):
                hits.append({
                    "id": doc_id,
                    "content": doc,
                    "metadata": meta,
                    "score": 1 - dist,  # convert cosine distance to similarity
                })

        return hits

    @property
    def count(self) -> int:
        return self._index.count()

import unittest
import tempfile


class TestVectorIndex(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.index = VectorIndex(index_name="test_index", storage_dir=self.tmp_dir)

    def _make_chunks(self, n: int) -> List[Document]:
        return [
            Document(
                page_content=f"This is chunk number {i}.",
                metadata={"source_file": "test.pdf", "page": i, "chunk_index": i}
            )
            for i in range(n)
        ]

    def _make_vectors(self, n: int, dim: int = 384) -> np.ndarray:
        return np.random.rand(n, dim).astype(np.float32)

    def test_add_and_search(self):
        chunks = self._make_chunks(5)
        vectors = self._make_vectors(5)
        self.index.add_chunks(chunks, vectors)

        self.assertEqual(self.index.count, 5)

        # NOTE: search correctness (semantic relevance) cannot be tested with random
        # vectors — that requires real embeddings and is verified in the notebook.
        hits = self.index.search(self._make_vectors(1)[0], top_k=3)
        self.assertEqual(len(hits), 3)
        self.assertIn("content", hits[0])
        self.assertIn("score", hits[0])

    def test_mismatched_chunks_and_vectors_raises_error(self):
        with self.assertRaises(ValueError):
            self.index.add_chunks(self._make_chunks(3), self._make_vectors(5))


if __name__ == "__main__":
    unittest.main()
