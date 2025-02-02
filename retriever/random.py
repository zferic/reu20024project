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
import random

DIR = "retriever"
class RandomRetriever(EmbeddingRetriever):


    def __init__(self, docs_path : str = "papers", vectors_path : str = "vectorstore", recreate : bool = False):
        super().__init__(docs_path, vectors_path, recreate)
        doc_ids = self.vector_store.index_to_docstore_id.values()
        self.documents = [self.vector_store.docstore.search(doc_id) for doc_id in doc_ids]
    

    def __call__(self, query : str, n : int) -> list[Document]:
        return random.choices(self.documents, k = n)