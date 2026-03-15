import re
from typing import Dict, Any

from groq_llm import GroqLLM
from retriever import Retriever


class RAGPipeline:
    """Combines the retriever and LLM to answer questions about the document."""

    def __init__(self, retriever: Retriever, llm: GroqLLM):
        self._retriever = retriever
        self._llm = llm

    def _find_best_excerpt(self, content: str, question: str, length: int = 200) -> str:
        query_words = set(re.sub(r"[^\w\s]", "", question.lower()).split())
        sentences = re.split(r"(?<=[.!?])\s+", content)

        best_idx = 0
        best_overlap = -1
        for i, sentence in enumerate(sentences):
            words = set(re.sub(r"[^\w\s]", "", sentence.lower()).split())
            overlap = len(words & query_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i

        start = content.find(sentences[best_idx])
        excerpt = content[max(0, start - 50): start + length]
        return excerpt.strip() + "..."

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
                "preview": self._find_best_excerpt(r["content"], question),
            }
            for r in results
        ]

        answer = self._llm.generate(question, context)
        confidence = max(r["score"] for r in results)

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
