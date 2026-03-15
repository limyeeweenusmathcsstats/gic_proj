from typing import Dict, Any

from groq_llm import GroqLLM
from retriever import Retriever


class RAGPipeline:
    """Combines the retriever and LLM to answer questions about the document."""

    def __init__(self, retriever: Retriever, llm: GroqLLM):
        self._retriever = retriever
        self._llm = llm

    def ask(self, question: str, top_k: int = 5, min_score: float = 0.2) -> Dict[str, Any]:
        """Retrieve relevant chunks and generate an answer with source info."""
        results = self._retriever.retrieve(question, top_k=top_k, min_score=min_score)

        if not results:
            return {
                "answer": "No relevant content found in the document for this question.",
                "sources": [],
                "confidence": 0.0,
            }

        context = "\n\n".join([r["content"] for r in results])

        sources = [
            {
                "page": r["page"],
                "filename": r["source_file"],
                "score": round(r["score"], 2),
                "preview": r["content"][:200] + "...",
            }
            for r in results
        ]

        answer = self._llm.generate(question, context)
        confidence = sum(r["score"] for r in results) / len(results)

        return {
            "answer": answer,
            "sources": sources,
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
        self.assertIn("confidence", output)

    def test_ask_no_results_returns_fallback(self):
        pipeline = self._make_pipeline([])
        output = pipeline.ask("Some question")
        self.assertEqual(output["confidence"], 0.0)
        self.assertEqual(output["sources"], [])


if __name__ == "__main__":
    unittest.main()
