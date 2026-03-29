import math
from typing import Dict, Any, List
from sentence_transformers.cross_encoder import CrossEncoder

from groq_llm import GroqLLM
from retriever import Retriever


class RAGPipeline:
    """Combines the retriever and LLM to answer questions about the document."""

    def __init__(self, retriever: Retriever, llm: GroqLLM):
        self._retriever = retriever
        self._llm = llm
        self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def _sigmoid(self, x: float) -> float:
        return 1 / (1 + math.exp(-x))

    def _rerank(self, question: str, results: List[Dict], top_k: int) -> List[Dict]:
        pairs = [[question, r["content"]] for r in results]
        scores = self._reranker.predict(pairs)
        for r, s in zip(results, scores):
            r["score"] = float(s)
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def ask(self, question: str, top_k: int = 5, min_score: float = 0.2) -> Dict[str, Any]:
        """Retrieve relevant chunks and generate an answer with source info."""
        candidates = self._retriever.retrieve(question, top_k=top_k * 3, min_score=min_score)
        results = self._rerank(question, candidates, top_k) if candidates else candidates

        chunks = [
            {
                "page": r["page"],
                "source_file": r["source_file"],
                "score": round(r["score"], 4),
                "content": r["content"],
            }
            for r in candidates
        ]

        if not results:
            return {
                "answer": "No relevant content found in the document for this question.",
                "sources": [],
                "chunks": chunks,
                "confidence": 0.0,
            }

        context = "\n\n".join([r["content"] for r in results])

        sources = [
            {
                "page": r["page"],
                "source_file": r["source_file"],
                "score": round(self._sigmoid(r["score"]), 4),
                "content": r["content"],
            }
            for r in results
        ]

        answer = self._llm.generate(question, context)
        confidence = self._sigmoid(max(r["score"] for r in results))

        return {
            "answer": answer,
            "sources": sources,
            "chunks": chunks,
            "confidence": round(confidence, 4),
        }


import unittest
from unittest.mock import MagicMock


class TestRAGPipeline(unittest.TestCase):

    def _make_pipeline(self, retriever_results):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = retriever_results

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "The answer is 42."

        return RAGPipeline(mock_retriever, mock_llm)

    def test_ask_returns_expected_keys(self):
        results = [
            {"content": "Some text about revenue.", "score": 0.85, "page": 3, "source_file": "google.pdf"},
        ]
        pipeline = self._make_pipeline(results)
        output = pipeline.ask("What was the revenue?")

        self.assertIn("answer", output)
        self.assertIn("sources", output)
        self.assertIn("chunks", output)
        self.assertIn("confidence", output)

    def test_ask_no_results_returns_fallback(self):
        pipeline = self._make_pipeline([])
        output = pipeline.ask("Some question")
        self.assertEqual(output["confidence"], 0.0)
        self.assertEqual(output["sources"], [])
        self.assertEqual(output["chunks"], [])


if __name__ == "__main__":
    unittest.main()
