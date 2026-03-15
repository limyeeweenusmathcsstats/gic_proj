
- **Problem framing** — how you interpreted and scoped the task
Choice of data - Initially I chose Google's annual report only to realise later that it falls short of 100 pages
So I pivoted to Microsoft's report

- **System design** — your architecture, major design choices, alternatives considered, and trade-offs
Why RAG?
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