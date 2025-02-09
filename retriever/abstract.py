from abc import ABC, abstractmethod
from langchain_core.documents import Document

class AbstractRetriever:


    @abstractmethod
    def __call__(self, query : str, n : int) -> list[Document]:
        """
        Prompts a retriever to fetch n documents based on the given query 
        """
        ...


    def format(context : list[Document]) -> str:
        """
        Formats a retrieved list of documents as a single string
        """
        return "\n\n".join([c.page_content for c in context])
