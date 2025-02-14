from abc import ABC, abstractmethod
from langchain_core.documents import Document

class AbstractRetriever:


    @abstractmethod
    def __call__(self, query : str, n : int) -> list[Document]:
        """
        Prompts a retriever to fetch n documents based on the given query 
        """
        ...

    