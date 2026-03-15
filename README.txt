
- **Problem framing** — how you interpreted and scoped the task
Choice of data - Initially I chose Google's annual report only to realise later that it falls short of 100 pages
So I pivoted to Microsoft's report

- **System design** — your architecture, major design choices, alternatives considered, and trade-offs
Why RAG?
Downsides of long context window:
If we directly pass 100+ pages of text document into a LLM, it is computational expensive for the LLM to work with such a huge amount of text

RAG:
RAG solves this problem by breaking down a huge document into small chunks whose similarities with the question can be calculated efficiently using cosine similarity.
Then only those small chunks that are relavant are passed to the LLM

Cross Encoder Strategy -
To ensure that evidence selected answers the query directly,
we first use a bi-encoder to encode the question and the context separately
Then using cosine similarity scores we calculate the similarity between the question and the context
The top 15 (question, context) pairs that are most similar then go through re-ranking by the cross encoder
To do so, the embedding of the question and context are concatenated together with a special character in the middle
Then they are passed into a transformer model such that the attention mechanism of the transformer can tend to token pairs from question and context
subequently the embedding outputted by the transformer model is passed into a MLP to produce a similarity score.This ensures a more precise computation of similarity at a higher computational cost hence only the top 15 results go through this

Chunking strategy of 200


- **Setup instructions** — how to install dependencies and configure the environment


- **How to run the system** — clear steps to ingest a document and ask questions


- **Assumptions and limitations** — what you chose to leave out of scope and why



- **Future improvements** — what you would do next with more time


IMPORTANT ASSUMPTIONS when running the code:

1) The format of the file uploaded is a PDF file

Before running the code,
One needs to get a free API key at console.groq.com

To run the code, type the following command
python main.py

Debugging Log
each file consists of the following:
a question asked, the response given by my AI 
and my hypothesis on what went wrong