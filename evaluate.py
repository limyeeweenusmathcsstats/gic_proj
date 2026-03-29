from pathlib import Path
from datetime import datetime

from main import build_index
from vector_index import VectorIndex
from embedder import DocumentEmbedder
from retriever import Retriever
from groq_llm import GroqLLM
from rag_pipeline import RAGPipeline
from retriever_evaluator import RetrieverEvaluator
from generator_evaluator import GeneratorEvaluator
from eval_dataset import EVAL_DATASET

PDF_PATH = "data/text_files/microsoft.pdf"
VECTOR_STORE_DIR = "data/vector_store"


def _build_pipeline(pdf_path: str) -> RAGPipeline:
    index_name = Path(pdf_path).stem + "_index"
    index = VectorIndex(index_name=index_name, storage_dir=VECTOR_STORE_DIR)
    if index.count == 0:
        embedder = build_index(pdf_path, index)
    else:
        embedder = DocumentEmbedder()
    retriever = Retriever(index, embedder)
    llm = GroqLLM()
    return RAGPipeline(retriever, llm)


def _save_report(retriever_results: dict, generator_results: dict) -> str:
    log_dir = Path("testing_log")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = log_dir / f"eval_report_{timestamp}.txt"

    sep = "=" * 60
    lines = [
        sep,
        "  EVALUATION REPORT",
        f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        sep,
        "",
        "RETRIEVER SUMMARY",
        "-----------------",
        f"Hit Rate @ {retriever_results['k']}  : {retriever_results['hit_rate']}",
        f"MRR      @ {retriever_results['k']}  : {retriever_results['mrr']}",
        f"NDCG     @ {retriever_results['k']}  : {retriever_results['ndcg']}",
        f"Queries            : {retriever_results['num_queries']}",
        "",
        "GENERATOR SUMMARY",
        "-----------------",
        f"Avg Faithfulness : {generator_results['avg_faithfulness']}",
        f"Avg Relevance    : {generator_results['avg_relevance']}",
    ]
    if "avg_correctness" in generator_results:
        lines.append(f"Avg Correctness  : {generator_results['avg_correctness']}")
    lines += ["", sep, "", "PER-QUERY RETRIEVER BREAKDOWN", "-----------------------------"]

    for pq in retriever_results["per_query"]:
        lines += [
            f"Q: {pq['question']}",
            f"   Relevant pages  : {pq['relevant_pages']}",
            f"   Retrieved pages : {pq['retrieved_pages']}",
            f"   Hit Rate        : {pq['hit_rate']}  |  MRR: {pq['reciprocal_rank']}  |  NDCG: {pq['ndcg']}",
            "",
        ]

    lines += [sep, "", "PER-QUERY GENERATOR BREAKDOWN", "-----------------------------"]
    for pq in generator_results["per_query"]:
        lines += [
            f"Q: {pq['question']}",
            f"   Answer          : {pq['answer'][:200]}",
            f"   Faithfulness    : {pq['faithfulness']}  — {pq['faithfulness_reason'][:120]}",
            f"   Relevance       : {pq['relevance']}  — {pq['relevance_reason'][:120]}",
        ]
        if "correctness" in pq:
            lines.append(f"   Correctness     : {pq['correctness']}  — {pq['correctness_reason'][:120]}")
        lines.append("")

    lines.append(sep)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)


def main():
    print(f"Loading pipeline for: {PDF_PATH}")
    pipeline = _build_pipeline(PDF_PATH)

    print(f"\nRunning retriever evaluation on {len(EVAL_DATASET)} questions...")
    ret_ev = RetrieverEvaluator(pipeline)
    retriever_results = ret_ev.evaluate(EVAL_DATASET, k=5)

    print(f"Running generator evaluation on {len(EVAL_DATASET)} questions...")
    judge = GroqLLM()
    gen_ev = GeneratorEvaluator(pipeline, judge)
    generator_results = gen_ev.evaluate(EVAL_DATASET)

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"\nRetriever  (k=5)")
    print(f"  Hit Rate : {retriever_results['hit_rate']}")
    print(f"  MRR      : {retriever_results['mrr']}")
    print(f"  NDCG     : {retriever_results['ndcg']}")
    print(f"\nGenerator")
    print(f"  Faithfulness : {generator_results['avg_faithfulness']}")
    print(f"  Relevance    : {generator_results['avg_relevance']}")
    if "avg_correctness" in generator_results:
        print(f"  Correctness  : {generator_results['avg_correctness']}")

    report_path = _save_report(retriever_results, generator_results)
    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    main()
