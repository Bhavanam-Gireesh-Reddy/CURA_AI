import os

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader
)


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


DATA_PATH = os.path.join(
    BASE_DIR,
    "data"
)


class FileLoader:

    def __init__(self, path=DATA_PATH):

        self.path = path
        self.documents = []


    def load_files(self):

        text_loader = DirectoryLoader(
            self.path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={
                "encoding":"utf-8"
            }
        )


        pdf_loader = DirectoryLoader(
            self.path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )


        docx_loader = DirectoryLoader(
            self.path,
            glob="**/*.docx",
            loader_cls=Docx2txtLoader
        )


        self.documents.extend(
            text_loader.load()
        )


        self.documents.extend(
            pdf_loader.load()
        )


        self.documents.extend(
            docx_loader.load()
        )


        print(
            "Documents Loaded:",
            len(self.documents)
        )


        return self.documents