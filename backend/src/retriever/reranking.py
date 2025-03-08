import os 
from typing import List, Tuple
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
        # 1. Initialize retriever components
        self.inner_retriever = inner_retriever
        self.first_pass_n = first_pass_n
        self.device = get_device()
        self.batch_size = batch_size
        self.cache = {}
        
        # 2. Load reranker model
        print(f"Loading reranker model on {self.device}")
        self.reranker = CrossEncoder(
            'cross-encoder/ms-marco-MiniLM-L-6-v2', 
            device=self.device,
            max_length=512
        )

    def _batch_predict(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        # 1. Process documents in batches
        all_pairs = []
        
        # 2. Process each batch
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i+self.batch_size]
            pairs = [(query, doc.page_content) for doc in batch]
            scores = self.reranker.predict(pairs)
            
            # 3. Combine documents with their scores
            doc_score_pairs = list(zip(batch, scores))
            all_pairs.extend(doc_score_pairs)
        
        # 4. Sort by score (descending)
        return sorted(all_pairs, key=lambda x: x[1], reverse=True)

    def __call__(self, query: str, n: int) -> List[Document]:
        # 1. Check cache first
        cache_key = f"{query}_{n}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 2. Get first pass results
        first_pass = self.inner_retriever(query, self.first_pass_n)
        
        # 3. Rerank in batches
        reranked = self._batch_predict(query, first_pass)
        
        # 4. Get top n results
        results = [doc for doc, _ in reranked][:n]
        
        # 5. Cache results
        self.cache[cache_key] = results
        
        return results