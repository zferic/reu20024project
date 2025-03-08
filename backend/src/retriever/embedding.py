import os 
import sys
import re
import time
from typing import Callable, List
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
import pickle
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredFileLoader
from langchain_core.documents import Document
import torch
import multiprocessing
from tqdm import tqdm
from .abstract import AbstractRetriever
from backend.utils.device import get_device

class EmbeddingRetriever(AbstractRetriever):
    def __init__(self, docs_path: str = "papers", vectors_path: str = "vectorstore", recreate: bool = False, batch_size: int = 32):
        self.vectors_path = vectors_path
        self.docs_path = docs_path
        self.device = get_device()
        self.batch_size = batch_size
        self.cache = {}  # Simple in-memory cache for query results
        
        if recreate or not os.path.exists(vectors_path):
            if not os.path.exists(docs_path):
                raise Exception(f"Could not find path {docs_path}, needed to initialize vector store")
            else:
                self._initialize_vector_store()
        
        try:
            print(f"Loading vector store from {vectors_path}")
            start_time = time.time()
            with open(vectors_path, "rb") as f:
                self.vector_store: FAISS = pickle.load(f)
            print(f"Vector store loaded in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            raise Exception(f"Failed to load vector store: {e}")

    def _deduplicate_chunks(self, chunks: list[Document]):
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

    def _process_batch(self, batch: List[Document], embeddings):
        """Process a batch of documents to create embeddings"""
        try:
            return FAISS.from_documents(batch, embeddings)
        except Exception as e:
            print(f"Error processing batch: {e}")
            # If batch fails, try processing one by one
            results = []
            for doc in batch:
                try:
                    single_result = FAISS.from_documents([doc], embeddings)
                    results.append(single_result)
                except Exception as inner_e:
                    print(f"Skipping document due to error: {inner_e}")
            
            # Merge results if any were successful
            if results:
                merged = results[0]
                for result in results[1:]:
                    merged.merge_from(result)
                return merged
            return None

    def _initialize_vector_store(self):
        """
        Creates the vector store at self.vectors_path using the documents specified at self.docs_path
        Uses batched processing and multiprocessing for better CPU efficiency
        """
        try:
            print(f"Initializing vector store from {self.docs_path}")
            start_time = time.time()
            
            # Load documents
            loader = DirectoryLoader(self.docs_path, glob="**/*.txt", loader_cls=UnstructuredFileLoader)
            documents = loader.load()
            print(f"Loaded {len(documents)} documents in {time.time() - start_time:.2f} seconds")
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = text_splitter.split_documents(documents)
            unique_splits = self._deduplicate_chunks(splits)
            print(f"Created {len(unique_splits)} unique chunks in {time.time() - start_time:.2f} seconds")
            
            # Initialize embeddings with optimized settings for CPU
            embeddings = HuggingFaceEmbeddings(
                model_name='sentence-transformers/all-distilroberta-v1', 
                model_kwargs={
                    'device': self.device,
                    # Add optimizations for CPU
                    'compute_dtype': torch.float32 if self.device == "cpu" else torch.float16,
                }
            )
            
            # Process in batches
            vectorstore = None
            batch_size = self.batch_size
            
            print(f"Processing {len(unique_splits)} documents in batches of {batch_size}")
            for i in tqdm(range(0, len(unique_splits), batch_size)):
                batch = unique_splits[i:i+batch_size]
                batch_result = self._process_batch(batch, embeddings)
                
                if batch_result:
                    if vectorstore is None:
                        vectorstore = batch_result
                    else:
                        vectorstore.merge_from(batch_result)
                
                # Save intermediate results periodically
                if i > 0 and i % (batch_size * 10) == 0:
                    print(f"Saving intermediate results after processing {i} documents")
                    with open(f"{self.vectors_path}_partial_{i}", "wb") as f:
                        pickle.dump(vectorstore, f)
            
            # Save final vector store
            print(f"Saving final vector store to {self.vectors_path}")
            with open(self.vectors_path, "wb") as f:
                pickle.dump(vectorstore, f)
                
            print(f"Vector store initialization completed in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            raise Exception(f"Failed to initialize vector store: {e}")

    def __call__(self, query: str, n: int) -> list[Document]:
        """
        Retrieves n documents from the vector store based on the given query
        Uses caching to avoid redundant searches
        """
        # Check cache first
        cache_key = f"{query}_{n}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Perform search
        results = self.vector_store.similarity_search(query, k=n)
        
        # Cache results
        self.cache[cache_key] = results
        
        return results