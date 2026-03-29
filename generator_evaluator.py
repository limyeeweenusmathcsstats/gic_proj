import re
from typing import Dict, Any, List

from langchain_core.messages import HumanMessage

from groq_llm import GroqLLM
from rag_pipeline import RAGPipeline


class GeneratorEvaluator:
    """Evaluates generator quality using LLM-as-judge.

    Each test case should have:
        - "question": the query string
        - "reference_answer" (optional): a known correct answer for correctness scoring

    Metrics (scored 1-5):
        - Faithfulness: is the answer grounded only in the retrieved context?
        - Relevance: does the answer actually address the question?
        - Correctness: does the answer convey the same facts as the reference answer?
    """

    def __init__(self, pipeline: RAGPipeline, judge_llm: GroqLLM):
        self._pipeline = pipeline
        self._judge = judge_llm

    def _ask_judge(self, prompt: str) -> str:
        response = self._judge.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def _parse_score(self, text: str) -> int:
        match = re.search(r"[1-5]", text)
        return int(match.group()) if match else 1

    def _score_faithfulness(self, question: str, context: str, answer: str) -> tuple[int, str]:
        prompt = f"""You are an expert judge evaluating a RAG system.

Context: {context}

Question: {question}
Answer: {answer}

Task: Is the Answer supported ONLY by the Context? (Faithfulness)
Score 1-5 (5 = perfectly faithful, 1 = total hallucination not in context).
Reply with your score on the first line, then a brief reason."""
        raw = self._ask_judge(prompt)
        return self._parse_score(raw), raw.strip()

    def _score_relevance(self, question: str, answer: str) -> tuple[int, str]:
        prompt = f"""You are an expert judge evaluating a RAG system.

Question: {question}
Answer: {answer}

Task: Does the Answer directly address the Question? (Relevance)
Score 1-5 (5 = directly and completely answers the question, 1 = completely off-topic).
Reply with your score on the first line, then a brief reason."""
        raw = self._ask_judge(prompt)
        return self._parse_score(raw), raw.strip()

    def _score_correctness(self, question: str, answer: str, reference: str) -> tuple[int, str]:
        prompt = f"""You are an expert judge evaluating a RAG system.

Question: {question}
Reference Answer: {reference}
Generated Answer: {answer}

Task: Does the Generated Answer convey the same key facts as the Reference Answer? (Correctness)
Score 1-5 (5 = all key facts match, 1 = factually wrong or completely missing).
Reply with your score on the first line, then a brief reason."""
        raw = self._ask_judge(prompt)
        return self._parse_score(raw), raw.strip()

    def evaluate(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not test_cases:
            raise ValueError("test_cases must not be empty.")

        faithfulness_scores, relevance_scores, correctness_scores = [], [], []
        per_query = []

        for tc in test_cases:
            question = tc["question"]
            reference = tc.get("reference_answer")
            result = self._pipeline.ask(question)
            answer = result["answer"]
            context = "\n\n".join(s["content"] for s in result["sources"])

            faith_score, faith_reason = self._score_faithfulness(question, context, answer)
            rel_score, rel_reason = self._score_relevance(question, answer)

            faithfulness_scores.append(faith_score)
            relevance_scores.append(rel_score)

            entry = {
                "question": question,
                "answer": answer,
                "faithfulness": faith_score,
                "faithfulness_reason": faith_reason,
                "relevance": rel_score,
                "relevance_reason": rel_reason,
            }

            if reference:
                corr_score, corr_reason = self._score_correctness(question, answer, reference)
                correctness_scores.append(corr_score)
                entry["correctness"] = corr_score
                entry["correctness_reason"] = corr_reason

            per_query.append(entry)

        n = len(test_cases)
        result = {
            "avg_faithfulness": round(sum(faithfulness_scores) / n, 4),
            "avg_relevance": round(sum(relevance_scores) / n, 4),
            "num_queries": n,
            "per_query": per_query,
        }
        if correctness_scores:
            result["avg_correctness"] = round(sum(correctness_scores) / len(correctness_scores), 4)
        return result


import unittest
from unittest.mock import MagicMock, patch


def _make_mocks(answer="The revenue was $100B.", faith_response="5\nPerfectly supported.", rel_response="4\nDirectly answers.", corr_response="5\nMatches reference."):
    pipeline = MagicMock(spec=RAGPipeline)
    pipeline.ask.return_value = {
        "answer": answer,
        "sources": [{"content": "Revenue was $100B in 2023.", "page": 3, "source_file": "doc.pdf", "score": 0.9}],
        "chunks": [],
        "confidence": 0.9,
    }
    judge = MagicMock(spec=GroqLLM)
    judge.llm = MagicMock()
    judge.llm.invoke.side_effect = [
        MagicMock(content=faith_response),
        MagicMock(content=rel_response),
        MagicMock(content=corr_response),
    ]
    return pipeline, judge


class TestGeneratorEvaluator(unittest.TestCase):

    def test_returns_expected_keys(self):
        pipeline, judge = _make_mocks()
        ev = GeneratorEvaluator(pipeline, judge)
        result = ev.evaluate([{"question": "What was the revenue?", "reference_answer": "Revenue was $100B."}])
        for key in ("avg_faithfulness", "avg_relevance", "avg_correctness", "num_queries", "per_query"):
            self.assertIn(key, result)

    def test_correctness_score_parsed(self):
        pipeline, judge = _make_mocks(corr_response="4\nMostly correct.")
        ev = GeneratorEvaluator(pipeline, judge)
        result = ev.evaluate([{"question": "What was the revenue?", "reference_answer": "Revenue was $100B."}])
        self.assertEqual(result["avg_correctness"], 4.0)

    def test_no_correctness_when_no_reference(self):
        pipeline, judge = _make_mocks()
        ev = GeneratorEvaluator(pipeline, judge)
        result = ev.evaluate([{"question": "What was the revenue?"}])
        self.assertNotIn("avg_correctness", result)
        self.assertNotIn("correctness", result["per_query"][0])

    def test_scores_parsed_correctly(self):
        pipeline, judge = _make_mocks(faith_response="5\nPerfect.", rel_response="3\nPartial.")
        ev = GeneratorEvaluator(pipeline, judge)
        result = ev.evaluate([{"question": "What was the revenue?"}])
        self.assertEqual(result["avg_faithfulness"], 5.0)
        self.assertEqual(result["avg_relevance"], 3.0)

    def test_fallback_score_when_no_digit(self):
        pipeline, judge = _make_mocks(faith_response="No score here.", rel_response="Also none.", corr_response="Nothing.")
        ev = GeneratorEvaluator(pipeline, judge)
        result = ev.evaluate([{"question": "What was the revenue?", "reference_answer": "$100B."}])
        self.assertEqual(result["avg_faithfulness"], 1.0)
        self.assertEqual(result["avg_relevance"], 1.0)
        self.assertEqual(result["avg_correctness"], 1.0)

    def test_aggregate_over_multiple_queries(self):
        pipeline = MagicMock(spec=RAGPipeline)
        pipeline.ask.return_value = {
            "answer": "Some answer.",
            "sources": [{"content": "Some context.", "page": 1, "source_file": "doc.pdf", "score": 0.8}],
            "chunks": [],
            "confidence": 0.8,
        }
        judge = MagicMock(spec=GroqLLM)
        judge.llm = MagicMock()
        judge.llm.invoke.side_effect = [
            MagicMock(content="5\nGood."), MagicMock(content="4\nGood."), MagicMock(content="5\nCorrect."),  # query 1
            MagicMock(content="3\nOk."),  MagicMock(content="2\nWeak."), MagicMock(content="3\nPartial."),   # query 2
        ]
        ev = GeneratorEvaluator(pipeline, judge)
        result = ev.evaluate([
            {"question": "q1", "reference_answer": "ref1"},
            {"question": "q2", "reference_answer": "ref2"},
        ])
        self.assertEqual(result["num_queries"], 2)
        self.assertEqual(result["avg_faithfulness"], 4.0)
        self.assertEqual(result["avg_relevance"], 3.0)
        self.assertEqual(result["avg_correctness"], 4.0)

    def test_empty_test_cases_raises(self):
        pipeline = MagicMock(spec=RAGPipeline)
        judge = MagicMock(spec=GroqLLM)
        ev = GeneratorEvaluator(pipeline, judge)
        with self.assertRaises(ValueError):
            ev.evaluate([])


if __name__ == "__main__":
    unittest.main()
