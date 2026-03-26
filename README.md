# RAG-Based Document Q&A System

## Table of Contents

1. [Problem Framing](#1-problem-framing)
2. [System Design](#2-system-design)
3. [Setup Instructions](#3-setup-instructions)
4. [How to Run the System](#4-how-to-run-the-system)
5. [Assumptions and Limitations](#5-assumptions-and-limitations)
6. [Testing and Results](#6-testing-and-results)
7. [Future Improvements](#7-future-improvements)
8. [Other Files of Interest](#8-other-files-of-interest)

---

## 1. Problem Framing

The problem to be solved in this project is to build an AI system capable of accurately answering questions when given a long document. I interpreted the task as a problem in retrieval-augmented question answering, where the AI has to answer based on information only available in the document and not in the training data of an LLM.

**Choice of data & LLM:**
To ensure that the LLM does not have prior knowledge of the document's contents, I chose the llama-3.3 model whose knowledge cut-off date is Dec 2023, as verifiable by running `python check_cutoff.py`. Additionally, I chose documents published after the cut-off date. Specifically, I initially chose Google's annual report but switched to Microsoft's annual report as the Google report fell short of 100 pages.

**Key aspects of the problem:**
- The program needs to handle a variety of data, from narrative paragraphs to structured tables and repeated themes.
- The solution should minimise hallucination. When the nature of a question is ambiguous, the AI should err on the side of caution rather than hallucinate.
- The approach should be domain-agnostic; a finance document was chosen as it satisfies the properties above.
- Focus on building a robust, understandable, and practical pipeline as opposed to research-level optimisations.

---

## 2. System Design

### Why RAG Instead of a Full Context Window?

**Downsides of passing the full document:**
Passing 100+ pages directly into an LLM is computationally expensive and strains the model's ability to attend to all content equally.

**Why RAG:**
RAG solves this by breaking the document into small chunks whose similarity to a query can be computed efficiently via cosine similarity. Only the most relevant chunks are passed to the LLM, which also reduces hallucination.

### Architecture Overview

The core pipeline components are:

1. Document ingestion and chunking
2. Embedding and indexing of document chunks
3. Retrieval of relevant chunks based on the user's query
4. Re-ranking of retrieved evidence
5. LLM-based answer generation using only the retrieved evidence

### Major Design Choices

**1. Chunking strategy (window size: 200)**
Long overlapping windows are used to avoid missing information at chunk boundaries.

**2. Embedding similarity search**
A bi-encoder model generates embeddings for both document chunks and user queries, enabling fast cosine similarity retrieval.

**3. Re-ranking: Cross-Encoder Strategy**
After the bi-encoder retrieves the top 15 candidate chunks, a cross-encoder re-ranks them. The question and context embeddings are concatenated with a separator token and fed into a transformer model, whose attention mechanism jointly attends to both. The resulting embedding is passed through an MLP to produce a refined similarity score. This yields more precise relevance estimates at a higher computational cost — hence it is applied only to the top 15 candidates.

### Alternatives Considered

| Alternative | Reason Rejected |
|---|---|
| Direct LLM ingestion | Context window and cost limitations |
| Classical IR (e.g. BM25) | Embedding-based retrieval provides better semantic matching |
| No cross-encoder re-ranking | Faster, but evidence may not directly answer the question |

### Trade-offs

**Accuracy vs. Efficiency:**
Cross-encoder re-ranking improves answer quality but increases latency. The system limits re-ranking to the top 15 candidates to balance both.

**Chunk length:**

| | Short Chunks | Long Chunks |
|---|---|---|
| **Pros** | Finds specific, targeted information | Preserves broader context; fewer chunks means faster retrieval |
| **Cons** | May lose context when answers span multiple chunks | Risk of retrieving chunks with irrelevant information alongside the answer |

---

## 3. Setup Instructions

1. Obtain a free API key at [console.groq.com](https://console.groq.com).
2. Create a `.env` file in the project root directory and add:
   ```
   GROQ_API_KEY=your_api_key_here
   ```
3. Set up a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 4. How to Run the System

```bash
python main.py
```

- When prompted to pick a document, type `2`.
- When prompted to enter a question, type your question.
- The answer will be printed to the terminal and logged to a text file in `testing_log/`.

---

## 5. Assumptions and Limitations

### Assumptions

- The input document is a single PDF file exceeding 100 pages, as required by the project instructions.
- The document is in English and contains a mix of narrative text, structured data, and tables, as is typical of finance or annual reports.
- The LLM (llama-3.3) has a knowledge cutoff before the document's publication date, ensuring no prior exposure to the test content.
- All user questions are answerable solely from the provided document, not from external knowledge.
- The user will provide a valid API key and follow the setup instructions.

### Limitations

- High-quality, well-grounded answers can receive low confidence scores if they require inference or draw from information scattered across multiple pages.
- RAG struggled with certain technical jargon (e.g. *pro forma* income vs. income).
- The system does not perform advanced document layout analysis (e.g. for complex tables or figures), so answers may be less accurate for highly structured or graphical content.
- The chunking and retrieval approach may miss answers that span multiple chunks or require deep cross-referencing.
- The LLM's responses are limited by the quality of retrieved evidence; if relevant content is not retrieved, the answer may be incomplete.

---

## 6. Testing and Results

The system was evaluated on 12 questions drawn from Microsoft's 2024 annual report. Questions and responses are saved in the `testing_log/` folder, named `microsoft_<question_summary>.txt`.

One question (the company's mission and vision) did not have its answer in the document — the program correctly declined to hallucinate an answer.

For the remaining 11 questions:

| Result | Count | Notes |
|---|---|---|
| Completely correct | 7 / 11 | Covered a wide range: quantitative (e.g. net income) and qualitative (e.g. legal proceedings) |
| Almost completely correct | 4 / 11 | Minor gaps due to ambiguous phrasing, fine-grained detail, or domain jargon; no hallucination observed |
| Wrong | 0 / 11 | — |

The 4 near-correct answers involved subjective terms like "main risks" (where the document does not explicitly rank risks) or specific financial jargon (e.g. *pro forma* income). In all cases, the AI was slightly over-cautious rather than hallucinating.

**Overall: 0 hallucinations across all 12 questions.**

---

## 7. Future Improvements

- Support for additional document formats (Word, Excel, CSV, etc.)
- Specialised parsing for complex tables and figures, integrated with the text pipeline
- Multi-document analysis
- A user interface for document upload and interactive Q&A
- Adaptive chunking strategies based on document structure (e.g. average sentence length)
- Evaluate the system on domains outside of finance

---

## 8. Other Files of Interest

The `testing_log/debugging_logs/` folder contains debugging logs for individual queries. Each file includes:

- The question asked
- The response given by the AI
- A hypothesis on what went wrong
