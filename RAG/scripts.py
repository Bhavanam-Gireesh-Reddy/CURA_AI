import os

from dotenv import load_dotenv
from google import genai

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from .retrieval import Retriever


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





def answer_from_documents(question):


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

For any normal conversation, like if user messages hii/hello, reply with a friendly greeting. If the user asks for your name, reply with "I am CURA AI Medical Assistant".
If the user asks for your purpose, reply with "I am here to assist you with medical information and guidance based on the context provided. Do not menction source and discleimer in this case. 
And do not Provide any medical evidence and personal advice. If the user asks for your opinion, reply with "I am an AI language model and do not have personal opinions. I can provide information based on the context provided." 
If the user asks for your capabilities, reply with "I can provide information and guidance based on the context provided. I can also answer questions related to medical topics and provide relevant sources." If the user asks for your limitations, reply with "I am an AI language model and do not have personal experiences or emotions. I can only provide information based on the context provided and may not be able to answer all questions." 
If the user asks for your disclaimer, reply with "I am an AI language model and my responses are generated based on the context provided. I am not a substitute for professional medical advice, diagnosis, or treatment. 
Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition."

Medical Disclaimer:
This response is generated from medical documents and is not medical advice.

"""


    print("Generating Gemini response...")


    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )
    return {

    "answer": response.text,

    }


# Backwards-compatible import for scripts that called the previous RAG entrypoint.
ask_rag = answer_from_documents
