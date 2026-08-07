import os

from dotenv import load_dotenv
from google import genai

from doc_loader import FileLoader
from chunking import Chunk
from vector_db import Vectorizer
from retrieval import Retriever

TOP_K = 5
Model = "gemini-3.1-flash-lite"

load_dotenv()


def main():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Add it to your .env file."
        )

    documents = FileLoader().load_files()

    chunks = Chunk(documents).chunking()

    vector_db = Vectorizer(chunks).vectorize()

    retriever = Retriever()

    client = genai.Client(api_key=api_key)

    question = input("\nAsk Any Question: ").strip()

    if question.lower() in ["exit", "quit"]:
        print("Stopped.")
        return

    retrieved_docs = retriever.TopKretriever(
        question,
        vector_db,
        TOP_K
    )

    if not retrieved_docs:
        print("No relevant documents found.")
        return

    context = "\n\n".join(
        [
            doc.page_content
            for doc in retrieved_docs
        ]
    )

    system_prompt = f"""
You are an intelligent Medical RAG Assistant.

Answer ONLY from the retrieved context.

User Question:
{question}

Retrieved Context:
{context}

Instructions:

1. Use only the retrieved context.
2. Do not use external knowledge.
3. Do not hallucinate.
4. If information is unavailable, say:
   "The requested information was not found in the retrieved medical documents."
5. Mention source documents.
6. Do not provide medical diagnosis.
7. Include a medical disclaimer.

Output Format:

Disease/Condition Identified:
<disease>

Answer:
<answer>

Key Findings:
- finding 1
- finding 2
- finding 3

Sources:
- source 1
- source 2

Medical Disclaimer:
This response is generated from retrieved medical documents and should not be considered medical advice.
"""

    response = client.models.generate_content(
        model=Model,
        contents=system_prompt
    )

    print("\nGemini Response:\n")
    print(response.text)


if __name__ == "__main__":
    main()