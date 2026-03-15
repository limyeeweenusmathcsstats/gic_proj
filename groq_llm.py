import os
from unittest.mock import MagicMock

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()


class GroqLLM:
    """Wraps the Groq API to generate answers from retrieved context."""

    def __init__(self, model_name: str = "llama-3.3-70b-versatile", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set.")

        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.1,
            max_tokens=1024,
        )

    def generate(self, query: str, context: str) -> str:
        """Generate ans."""
        prompt = (
            "You are a helpful assistant. Use only the context below to answer the question.\n"
            "If the context does not contain enough information, say so — do not make up an answer.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content


import unittest
from unittest.mock import MagicMock


class TestGroqLLM(unittest.TestCase):

    def test_raises_if_no_api_key(self):
        os.environ.pop("GROQ_API_KEY", None)
        with self.assertRaises(ValueError):
            GroqLLM(api_key=None)

    def test_generate_returns_string(self):
        llm = GroqLLM.__new__(GroqLLM)
        llm.llm = MagicMock()
        llm.llm.invoke.return_value = MagicMock(content="some answer")

        result = llm.generate("What was Google's revenue?", "Revenue was $307 billion in 2023.")
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
