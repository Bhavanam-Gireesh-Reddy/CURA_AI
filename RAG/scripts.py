import os

from dotenv import load_dotenv
from google import genai

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from .retrieval import Retriever
from .tools import liver_tool, xray_tool, mri_tool


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


CHROMA_DIR = os.path.join(
    BASE_DIR,
    "chromadb"
)


load_dotenv()


TOP_K = 5

MODEL = "gemini-3.1-flash-lite"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


vector_db = None
client = None
embeddings = None

retriever = Retriever()



def load_rag():

    global vector_db
    global client
    global embeddings


    if vector_db is not None:
        return


    print("1. Loading RAG System...")


    api_key = os.getenv("GEMINI_API_KEY")


    if not api_key:
        raise Exception(
            "Gemini API Key Missing"
        )


    print("2. API key loaded")


    client = genai.Client(
        api_key=api_key
    )


    print("3. Gemini loaded")


    print("4. Loading embedding model")


    try:

        embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={
        "device": "cpu"
    }
)

    except Exception as e:

        print("Embedding Error:", e)

        raise e



    print("5. Embedding loaded")


    vector_db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )


    print("6. Chroma loaded")


    print("RAG System Loaded Successfully!")





def ask_rag(question):


    load_rag()


    print("Searching documents...")


    docs = retriever.TopKretriever(
        question,
        vector_db,
        TOP_K
    )


    if not docs:

        return {
            "answer":
            "No relevant information found."
        }



    context = "\n\n".join(
        [
            doc.page_content
            for doc in docs
        ]
    )


    prompt = f"""

You are CURA AI Medical Assistant.

Answer only using the context.

Question:
{question}


Context:
{context}


Provide:

Answer:

Sources:

Medical Disclaimer:
This response is generated from medical documents and is not medical advice.

"""


    print("Generating Gemini response...")


    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )
    tool = detect_tool(question)


    return {

    "answer": response.text,

    "tool": tool

    }


def detect_tool(question):

    question = question.lower()


    if "liver" in question or "fatty liver" in question:
        return liver_tool()


    elif "fracture" in question or "xray" in question or "x-ray" in question:
        return xray_tool()


    elif "mri" in question or "brain tumor" in question:
        return mri_tool()


    return None