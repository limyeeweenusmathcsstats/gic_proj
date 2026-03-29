import math
from typing import Dict, Any, List

from rag_pipeline import RAGPipeline


class RetrieverEvaluator:
    """Evaluates retriever quality using Hit Rate, MRR, and NDCG.

    Each test case should have:
        - "question": the query string
        - "relevant_pages": list of 1-based page numbers that are ground truth
    """

    def __init__(self, pipeline: RAGPipeline):
        self._pipeline = pipeline

    def _hit_rate(self, retrieved_pages: List[int], relevant_pages: List[int], k: int) -> float:
        relevant_set = set(relevant_pages)
        return 1.0 if any(p in relevant_set for p in retrieved_pages[:k]) else 0.0

    def _reciprocal_rank(self, retrieved_pages: List[int], relevant_pages: List[int], k: int) -> float:
        relevant_set = set(relevant_pages)
        for rank, page in enumerate(retrieved_pages[:k], start=1):
            if page in relevant_set:
                return 1.0 / rank
        return 0.0

    def _ndcg(self, retrieved_pages: List[int], relevant_pages: List[int], k: int) -> float:
        relevant_set = set(relevant_pages)
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, page in enumerate(retrieved_pages[:k], start=1)
            if page in relevant_set
        )
        ideal_hits = min(len(relevant_set), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        return dcg / idcg if idcg > 0 else 0.0

    def evaluate(self, test_cases: List[Dict[str, Any]], k: int = 5) -> Dict[str, Any]:
        if not test_cases:
            raise ValueError("test_cases must not be empty.")

        hit_rates, rrs, ndcgs = [], [], []
        per_query = []

        for tc in test_cases:
            question = tc["question"]
            relevant_pages = tc["relevant_pages"]

            result = self._pipeline.ask(question)
            retrieved_pages = [c["page"] for c in result["chunks"]]

            hr = self._hit_rate(retrieved_pages, relevant_pages, k)
            rr = self._reciprocal_rank(retrieved_pages, relevant_pages, k)
            ndcg = self._ndcg(retrieved_pages, relevant_pages, k)

            hit_rates.append(hr)
            rrs.append(rr)
            ndcgs.append(ndcg)

            per_query.append({
                "question": question,
                "relevant_pages": relevant_pages,
                "retrieved_pages": retrieved_pages,
                "hit_rate": hr,
                "reciprocal_rank": round(rr, 4),
                "ndcg": round(ndcg, 4),
            })

        n = len(test_cases)
        return {
            "hit_rate": round(sum(hit_rates) / n, 4),
            "mrr": round(sum(rrs) / n, 4),
            "ndcg": round(sum(ndcgs) / n, 4),
            "k": k,
            "num_queries": n,
            "per_query": per_query,
        }


import unittest
from unittest.mock import MagicMock


def _make_pipeline(chunks):
    pipeline = MagicMock(spec=RAGPipeline)
    pipeline.ask.return_value = {
        "answer": "Some answer.",
        "sources": [],
        "chunks": chunks,
        "confidence": 0.9,
    }
    return pipeline


def _chunks(pages):
    return [{"page": p, "source_file": "doc.pdf", "score": 0.9, "content": f"text p{p}"} for p in pages]


class TestRetrieverEvaluator(unittest.TestCase):

    def test_hit_rate_found(self):
        ev = RetrieverEvaluator(_make_pipeline(_chunks([3, 5, 7])))
        result = ev.evaluate([{"question": "q", "relevant_pages": [5]}], k=3)
        self.assertEqual(result["hit_rate"], 1.0)

    def test_hit_rate_not_found(self):
        ev = RetrieverEvaluator(_make_pipeline(_chunks([1, 2, 3])))
        result = ev.evaluate([{"question": "q", "relevant_pages": [99]}], k=5)
        self.assertEqual(result["hit_rate"], 0.0)

    def test_mrr_first_rank(self):
        ev = RetrieverEvaluator(_make_pipeline(_chunks([5, 3, 7])))
        result = ev.evaluate([{"question": "q", "relevant_pages": [5]}], k=5)
        self.assertAlmostEqual(result["mrr"], 1.0)

    def test_mrr_second_rank(self):
        ev = RetrieverEvaluator(_make_pipeline(_chunks([1, 5, 3])))
        result = ev.evaluate([{"question": "q", "relevant_pages": [5]}], k=5)
        self.assertAlmostEqual(result["mrr"], 0.5)

    def test_ndcg_perfect(self):
        ev = RetrieverEvaluator(_make_pipeline(_chunks([5, 3, 7])))
        result = ev.evaluate([{"question": "q", "relevant_pages": [5]}], k=3)
        self.assertAlmostEqual(result["ndcg"], 1.0)

    def test_aggregate_over_multiple_queries(self):
        pipeline = MagicMock(spec=RAGPipeline)
        pipeline.ask.side_effect = [
            {"answer": "a1", "sources": [], "chunks": _chunks([5, 3, 7]), "confidence": 0.9},
            {"answer": "a2", "sources": [], "chunks": _chunks([1, 2, 3]), "confidence": 0.1},
        ]
        ev = RetrieverEvaluator(pipeline)
        result = ev.evaluate(
            [{"question": "q1", "relevant_pages": [5]}, {"question": "q2", "relevant_pages": [99]}],
            k=5,
        )
        self.assertEqual(result["hit_rate"], 0.5)
        self.assertEqual(result["mrr"], 0.5)
        self.assertEqual(result["num_queries"], 2)


if __name__ == "__main__":
    unittest.main()
