# class Retriever:

#     def __init__(self):
#         pass

#     def TopKretriever(self, question, db, TOP_K):
#         retriever = db.as_retriever(search_kwargs = {"k":TOP_K})
#         retriever_doc = retriever.invoke(question)
#         context = "\n\n".join(doc.page_content for doc in retriever_doc)
#         print("Context of TOPKRetriever: \n", context)

#         return context

class Retriever:

    def __init__(self):
        pass

    def TopKretriever(self, question, db, TOP_K):
        retriever = db.as_retriever(
            search_kwargs={"k": TOP_K}
        )

        retrieved_docs = retriever.invoke(question)

        print("\nRetrieved Documents Count:", len(retrieved_docs))

        for i, doc in enumerate(retrieved_docs, start=1):
            source = doc.metadata.get("source", "Unknown Source")
            print(f"\nDocument {i} Source:", source)
            print(doc.page_content[:300])

        return retrieved_docs
