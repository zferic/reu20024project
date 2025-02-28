import os 
from typing import List
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from .embedding import EmbeddingRetriever
from .abstract import AbstractRetriever
from backend.utils.device import get_device

class RerankingRetriever(AbstractRetriever):
    """
    Retriever which first uses an inner retriever, fetching `first_pass_n` number of documents (which is passed in through the constructor), before 
    reranking and returning the specified number of final documents when called
    """

    def __init__(self, inner_retriever: AbstractRetriever, first_pass_n: int):
        self.inner_retriever = inner_retriever
        self.first_pass_n = first_pass_n
        self.device = get_device()
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=self.device)

    def __call__(self, query: str, n: int) -> List[Document]:
        """
        Retrieves self.first_pass_n documents from the vector store, reranks, and finally
        returns the top n documents
        """
        first_pass = self.inner_retriever(query, self.first_pass_n)
        scores = self.reranker.predict([(query, doc.page_content) for doc in first_pass])
        reranked = sorted(zip(first_pass, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in reranked][:n]