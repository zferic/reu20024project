import os 
import sys
import re
from typing import Callable
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
import pickle
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredFileLoader
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import torch
sys.path.append("./")
from retriever.embedding import EmbeddingRetriever
from retriever.abstract import AbstractRetriever
from utils.device import get_device

DIR = "retriever"
class RerankingRetriever(EmbeddingRetriever):

    """
    Retriever which first uses an inner retriever, fetching `first_pass_n` number of documents (which is passed in through the constructor), before 
    reranking and returning the specified number of final documents when called
    """

    def __init__(self, inner_retriever : AbstractRetriever, first_pass_n : int, docs_path : str = "papers", vectors_path : str = "vectorstore", recreate : bool = False):
        super().__init__(docs_path, vectors_path, recreate)
        self.inner_retriever = inner_retriever
        self.first_pass_n = first_pass_n
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    

    def __call__(self, query : str, n : int) -> list[Document]:
        """
        Retrieves self.first_pass_n documents from the vector store, reranks, and finally
        returns the top n documents
        """
        first_pass = self.inner_retriever(query, self.first_pass_n)
        scores = self.reranker.predict([(query, doc.page_content) for doc in first_pass])
        reranked = sorted(zip(first_pass, scores), key = lambda x : x[1], reverse = True)
        return [doc for doc, _ in reranked][:n]