# Table of Contents

1. Problem Framing
2. System Design
3. Setup Instructions
4. How to Run the System
5. Assumptions and Limitations
6. Testing and Results
7. Future Improvements
8. Other Files of Interest

- **Problem framing** — how you interpreted and scoped the task

The problem to be solved in this project is to build an AI system capable of accurately answering questions when given a long document.
I interpreted the task as a problem in retrieval-augmented question answering, where the AI has to answer based on information only available in the document and not in the training data of an LLM.
Choice of data & LLM:
To ensure that the LLM does not have prior knowledge of the contents of the documents, I chose the llama-3.3 model whose cut-off date is Dec 2023, as verifiable by running python check_cutoff.py.
Additionally, I chose documents that were published after the cut-off date. Specifically, I chose Google's annual report initially but changed to Microsoft's annual report as it fell short of 100 pages.
Key aspects of the problem:
a) The program needs to handle a variety of data, from narrative paragraphs to structured tables and repeated themes.
b) The solution should minimise hallucination. When the nature of the question is ambiguous, the AI should err on the side of caution and provide a conservative answer as opposed to hallucinating.
c) The approach should be domain-agnostic, and a finance document is chosen as it satisfies the properties laid out in (a).
d) Focus on building a robust, understandable, and practical pipeline as opposed to research-level optimisations.


- **System design** — your architecture, major design choices, alternatives considered, and trade-offs
System design — your architecture, major design choices, alternatives considered, and trade-offs

1) Why RAG as opposed to a context window?
Downsides of long context window:
If we directly pass 100+ pages of a text document into an LLM, it is computationally expensive for the LLM to work with such a huge amount of text.
RAG:
RAG solves this problem by breaking down a huge document into small chunks whose similarities with the question can be calculated efficiently using cosine similarity.
Then only those small chunks that are relevant are passed to the LLM.
This also reduces hallucination.
Architecture overview of RAG:
The core components are document ingestion and chunking, embedding and indexing of document chunks, retrieval of relevant chunks based on user queries, re-ranking of evidence, and LLM-based answer generation using only retrieved evidence.
Major design choices:
1) Chunking strategy of 200
Long overlapping windows are used to avoid missing information at chunk boundaries.
2) Embedding similarity search
A bi-encoder model is used to generate embeddings for both document chunks and user queries.
3) Ranking the relevance of evidence: Cross-Encoder Strategy
To ensure that the evidence selected answers the query directly, we first use a bi-encoder to encode the question and the context separately. Then, using cosine similarity scores, we calculate the similarity between the question and the context. The top 15 (question, context) pairs that are most similar then go through re-ranking by the cross-encoder. To do so, the embeddings of the question and context are concatenated together with a special character in the middle. Then they are passed into a transformer model such that the attention mechanism of the transformer can attend to token pairs from the question and context. Subsequently, the embedding outputted by the transformer model is passed into an MLP to produce a similarity score. This ensures a more precise computation of similarity at a higher computational cost, hence only the top 15 results go through this.
Alternatives considered:

Direct LLM ingestion — rejected due to context window and cost limitations.
Classical IR — rejected as modern embedding-based retrieval provides better semantic matching.
No re-ranking with cross-encoder strategy — faster, but the evidence selected might not answer the question.

Trade-offs:
Accuracy vs. Efficiency:
Cross-encoder re-ranking improves answer quality but increases latency. The system balances this by limiting re-ranking to the top 15 candidates.
Length of chunks:
Small length.
Pros:
a) Helps the program find very specific and relevant pieces of information.
Cons:
a) May lose important context if the answer spans multiple chunks, making it harder for the LLM to generate a complete answer.
b) Increases the total number of chunks, causing retrieval to be slower and require more storage.
Longer chunks.
Pros:
a) Preserves more context, which can help the LLM understand the broader meaning and relationships in the text.
b) Reduces the number of chunks, making retrieval faster and less resource-intensive.
Cons:
a) Increases the risk of retrieving chunks that contain a lot of irrelevant information (mitigated by cross-encoder ranking), which can confuse the LLM or dilute the answer (mitigated by outputting confidence scores).


- **Setup instructions** — how to install dependencies and configure the environment

Before running the code,
One needs to get a free API key at console.groq.com

Create a .env file in the main directory and type in GROQ_API_KEY=your API key
Setup a virtual environment and download dependencies in requirements.txt

- **How to run the system** — clear steps to ingest a document and ask questions
To run the code, type the following command
python main.py

When prompted to pick the document, type 2.
When prompted to type the question, enter the question. Answer will be printed and logged to a text file.


- **Assumptions and limitations** — what you chose to leave out of scope and why

Assumptions:
a) The input document is a single PDF file exceeding 100 pages, as required by the project instructions.
b) The document is in English and contains a mix of narrative, structured data, and tables, as is typical for finance or annual reports.
c) The LLM (llama-3.3) has a knowledge cutoff before the document's publication date, ensuring no prior exposure to the test content.
d) All user questions are assumed to be answerable based solely on the provided document, not on external knowledge.
e) The user will provide a valid API key and follow setup instructions for environment configuration.

Limitations:
a) During my testing, I found that high quality well grounded answers can have low confidence scores if the answers require some inference or are scattered across multiple pages
b) RAG struggled with some technical jargon eg. pro forma income vs income
c) The system does not perform advanced document layout analysis (e.g., for complex tables or figures), so answers may be less accurate for highly structured or graphical content.
d) The chunking and retrieval approach may miss answers that span multiple chunks or require deep cross-referencing.
e) The LLM’s responses are limited by the quality of retrieved evidence; if relevant content is not retrieved, the answer may be incomplete or state that information is missing.


- **Tesing and results**
The system was evaluated on 12 questions drawn from Microsoft's 2024 annual report. The questions and responses can be found in the `testing_log` folder, with files named as `microsoft_question`.

Out of the 12 questions, 1 question did not have its answer in the document (specifically, the company's mission and vision), and the program was able to avoid hallucinating for that question.

For the remaining 11 questions:
The AI program got **7/11 completely correct** with minimal limitations. These questions required a wide range of different answer types, from quantitative factual ones (e.g. Microsoft's net income) to qualitative summaries spanning multiple pages (e.g. Microsoft's legal proceedings).
The AI program got **4/11 almost completely correct** with some limitations. These answers occurred for questions with ambiguous phrasing, for example "main risks faced by Microsoft." The word "main" is a subjective adjective that implies a ranked selection, yet the document does not explicitly rank or label any risks as more important than others. As a result, the AI was perhaps slightly over-cautious, acknowledging the subjectivity of the term rather than committing to a definitive list. Other questions required much finer-grained detail (e.g. R&D spending breakdown) or an understanding of specific jargon (e.g. pro forma income). Nonetheless, in all of these instances, there was no evidence of hallucination by the AI program.

There were **0/11 wrong answers**, demonstrating robustness against hallucination.

- **Future improvements** — what you would do next with more time
a) Support for more document formats eg. Word, Excel, CSV etc
b) Integrate specialised parsing for complex tables and figures and integrate that info with the text information
c) Multi-document analysis
d) A user interface for users to ask questions and upload documents
e) Experiment with different chunking strategies adapted to document nature eg. average length of a sentence in the document
f) Test the program on other domains outside of finance


Other files that might be of interest: Debugging Log
Each file consists of the following:
a question asked, the response given by my AI 
and my hypothesis on what went wrong