import os 
import time
from typing import List, Dict, Tuple
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

    def __init__(self, inner_retriever: AbstractRetriever, first_pass_n: int, batch_size: int = 8):
        self.inner_retriever = inner_retriever
        self.first_pass_n = first_pass_n
        self.device = get_device()
        self.batch_size = batch_size
        self.cache: Dict[str, List[Document]] = {}  # Cache for query results
        
        # Load reranker with optimized settings for CPU
        print(f"Loading reranker model on {self.device}")
        start_time = time.time()
        self.reranker = CrossEncoder(
            'cross-encoder/ms-marco-MiniLM-L-6-v2', 
            device=self.device,
            # Add optimizations for CPU
            max_length=512,  # Limit max length to reduce memory usage
        )
        print(f"Reranker model loaded in {time.time() - start_time:.2f} seconds")

    def _batch_predict(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """
        Process documents in batches to reduce memory usage
        """
        all_pairs = []
        all_scores = []
        
        # Process in batches
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i+self.batch_size]
            pairs = [(query, doc.page_content) for doc in batch]
            scores = self.reranker.predict(pairs)
            
            # Combine documents with their scores
            doc_score_pairs = list(zip(batch, scores))
            all_pairs.extend(doc_score_pairs)
        
        # Sort by score (descending)
        return sorted(all_pairs, key=lambda x: x[1], reverse=True)

    def __call__(self, query: str, n: int) -> List[Document]:
        """
        Retrieves self.first_pass_n documents from the vector store, reranks, and finally
        returns the top n documents
        """
        # Check cache first
        cache_key = f"{query}_{n}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Get first pass results
        start_time = time.time()
        first_pass = self.inner_retriever(query, self.first_pass_n)
        print(f"First pass retrieval completed in {time.time() - start_time:.2f} seconds")
        
        # Rerank in batches
        start_time = time.time()
        reranked = self._batch_predict(query, first_pass)
        print(f"Reranking completed in {time.time() - start_time:.2f} seconds")
        
        # Get top n results
        results = [doc for doc, _ in reranked][:n]
        
        # Cache results
        self.cache[cache_key] = results
        
        return results