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
from retriever.abstract import AbstractRetriever
from utils.device import get_device

DIR = "retriever"
class EmbeddingRetriever(AbstractRetriever):

    def __init__(self, docs_path : str = "papers", vectors_path : str = "vectorstore", recreate : bool = False):
        files = os.listdir("retriever")
        self.vectors_path = vectors_path
        self.docs_path = docs_path
        self.device = get_device()
        if recreate or self.vectors_path not in files:
            if self.docs_path not in files:
                raise Exception(f"Could not find path {self.docs_path} in ./{DIR}, needed to initialize vector store")
            else:
                self._initialize_vector_store()
        with open(f"{DIR}/{self.vectors_path}", "rb") as f:
            self.vector_store : FAISS = pickle.load(f)

    def _deduplicate_chunks(self, chunks : list[Document]):
        """
        Deduplicate document chunks based on their content.
        """
        seen = set()
        unique_chunks = []
        for chunk in chunks:
            if chunk.page_content not in seen:
                seen.add(chunk.page_content)
                unique_chunks.append(chunk)
        return unique_chunks

    def _initialize_vector_store(self):
        """
        Creates the vector store at self.vectors_path using the documents specified at self.docs_path
        """
        try:
            print("initializing")
            loader = DirectoryLoader(f"{DIR}/{self.docs_path}", glob="**/*.txt", loader_cls=UnstructuredFileLoader)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = text_splitter.split_documents(documents)
            unique_splits = self._deduplicate_chunks(splits)
            embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-distilroberta-v1', model_kwargs = {'device': self.device})
            vectorstore = FAISS.from_documents(unique_splits, embeddings)
            pickle.dump(vectorstore, open(self.vectors_path, "wb"))
        except Exception as e:
            raise Exception(f"Failed to initialize vector store: {e}")

    def __call__(self, query : str, n : int) -> list[Document]:
        """
        Retrieves n documents from the vector store based on the given query
        """
        return self.vector_store.similarity_search(query, k=n)