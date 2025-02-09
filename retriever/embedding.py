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
from logging import Logger
from typing import Optional
from dummy_logger import DummyLogger

DIR = "retriever"


    
class EmbeddingRetriever(AbstractRetriever):

    def __init__(self, docs_path : str = "papers", vectors_path : str = "vectorstore", recreate : bool = False, embedding_model_name = 'sentence-transformers/all-distilroberta-v1', logger : Optional[Logger] = None):
        self.embeddings = HuggingFaceEmbeddings(model_name = embedding_model_name)
        files = os.listdir("retriever")
        self.vectors_path = vectors_path
        self.docs_path = docs_path
        self.logger = logger if logger != None else DummyLogger(__name__)
        self.device = get_device()
        if recreate or self.vectors_path not in files:
            if self.docs_path not in files:
                raise Exception(f"Could not find path {self.docs_path} in ./{DIR}, needed to initialize vector store")
            else:
                self._initialize_vector_store()
        try:
            if os.path.exists(self.vectors_path):
                self._load_vector_store()
            else:
                self._initialize_vector_store()
        except Exception as e:
            self.logger.error(f"Failed to initialize vector store: {e}")
            raise Exception(f"Failed to initialize vector store: {e}")


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
    
    def _load_vector_store(self):
        self.logger.info("Loading existing FAISS index...")
        self.vector_store = FAISS.load_local(
            self.vectors_path, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def _initialize_vector_store(self):
        self.logger.info("Creating new FAISS index...")
        loader = DirectoryLoader(f"{DIR}/{self.docs_path}", glob="**/*.txt", loader_cls=UnstructuredFileLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(documents)
        unique_splits = self._deduplicate_chunks(splits)

        self.logger.info("Indexing documents...")
        self.vector_store = FAISS.from_documents(unique_splits, self.embeddings)
        self.vector_store.save_local(self.vectors_path)
        self.logger.info("FAISS vector store ready.")



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