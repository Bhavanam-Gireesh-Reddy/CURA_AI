import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)


# cura_ai folder path
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


CHROMA_DIR = os.path.join(
    BASE_DIR,
    "chromadb"
)



class Vectorizer:

    def __init__(self, chunks):
        self.chunks = chunks


    def vectorize(self):

        if Path(CHROMA_DIR).exists():

            print("Using Existing Vector DB...")

            db = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embeddings,
            )


        else:

            print("Creating New Vector DB...")

            db = Chroma.from_documents(
                self.chunks,
                embeddings,
                persist_directory=CHROMA_DIR,
            )


        print(
            "Vector DB Loaded Successfully!"
        )

        return db