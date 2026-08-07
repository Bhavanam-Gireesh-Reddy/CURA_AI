from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(
    model_name = EMBEDDING_MODEL
)

class Chunk:

    def __init__(self, documents):
        self.documents = documents

    def chunking(self):

        text_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
        )

        chunks = text_splitter.split_documents(self.documents)

        print("Chunks Created Sucessfully! ", len(chunks))

        return chunks