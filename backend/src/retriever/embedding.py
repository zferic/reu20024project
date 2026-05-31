import os 
import sys
import re
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
import pickle
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_core.documents import Document
import torch
from tqdm import tqdm
from .abstract import AbstractRetriever
from backend.utils.device import get_device

import logging
logger = logging.getLogger(__name__)


SECTION_REGEX = re.compile(r"(###\s+[A-Za-z ].+?\s+###)")

DROP_SECTIONS = {
    "References",
    "Funding",
    "Conflicts of Interest",
    "Publisher's Disclaimer",
    "Supplementary Materials",
    "Acknowledgements"
}

def split_by_sections(text: str):
    """
    Returns list of (section_name, section_text)
    """
    parts = SECTION_REGEX.split(text)
    sections = []

    # If no headers found, treat entire doc as one section
    if len(parts) == 1:
        return [("Body", text)]

    for i in range(1, len(parts), 2):
        header = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""

        section_name = header.replace("#", "").strip()
        sections.append((section_name, content.strip()))

    return sections

def get_splitter_for_section(section: str):
    section = section.lower()

    separators = ["\n\n", "\n", ". ", " ", ""]

    # 🔹 DATA DICTIONARY: do NOT split
    if "data_dictionary" in section or "dictionary" in section:
        logger.info(f"Splitting dictionary")
        return RecursiveCharacterTextSplitter(
            chunk_size=10_000,   # large enough to keep one block
            chunk_overlap=0,
            separators=["\n\n"],  # split only if truly massive
        )


    if "abstract" in section:
        return RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=50,
            separators=separators,
        )

    if "methods" in section:
        return RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=100,
            separators=separators,
        )

    if "results" in section:
        return RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=75,
            separators=separators,
        )

    if "discussion" in section:
        return RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=75,
            separators=separators,
        )

    if "conclusion" in section:
        return RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=50,
            separators=separators,
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=50,
        separators=separators,
    )

class EmbeddingRetriever(AbstractRetriever):
    def __init__(self, docs_path: str = "papers", vectors_path: str = "vectorstore", recreate: bool = False, batch_size: int = 32):
        # 1. Initialize paths and settings
        self.vectors_path = vectors_path
        self.docs_path = docs_path
        self.device = get_device()
        self.batch_size = batch_size
        self.cache = {}

   
        print("docs_path =", docs_path, flush=True)

        
        # 2. Create vector store if needed or load existing one
        if recreate or not os.path.exists(vectors_path):
            if not os.path.exists(docs_path):
                raise Exception(f"Could not find path {docs_path}, needed to initialize vector store")
            else:
                self._initialize_vector_store()
        
        # 3. Load the vector store
        try:
            print(f"Loading vector store from {vectors_path}")
            with open(vectors_path, "rb") as f:
                self.vector_store: FAISS = pickle.load(f)
            
            # 4. Fix compatibility issues with older transformers versions
            if hasattr(self.vector_store, 'embeddings'):
                embeddings = self.vector_store.embeddings
                if hasattr(embeddings, 'client') and hasattr(embeddings.client, '_modules'):
                    model = embeddings.client
                    if hasattr(model, 'config'):
                        config = model.config
                        # Add missing attributes for newer transformers versions
                        if not hasattr(config, '_output_attentions'):
                            config._output_attentions = False
                        if not hasattr(config, '_output_hidden_states'):
                            config._output_hidden_states = False
                        if not hasattr(config, '_use_cache'):
                            config._use_cache = True
        except Exception as e:
            raise Exception(f"Failed to load vector store: {e}")

    def _deduplicate_chunks(self, chunks: list[Document]):
        # 1. Remove duplicate chunks based on content
        seen = set()
        unique_chunks = []
        for chunk in chunks:
            if chunk.page_content not in seen:
                seen.add(chunk.page_content)
                unique_chunks.append(chunk)
        return unique_chunks

    def _process_batch(self, batch: List[Document], embeddings):
        # 1. Process a batch of documents to create embeddings
        try:
            return FAISS.from_documents(batch, embeddings)
        except Exception as e:
            print(f"Error processing batch: {e}")
            # 2. If batch fails, try processing one by one
            results = []
            for doc in batch:
                try:
                    single_result = FAISS.from_documents([doc], embeddings)
                    results.append(single_result)
                except Exception as inner_e:
                    print(f"Skipping document due to error: {inner_e}")
            
            # 3. Merge results if any were successful
            if results:
                merged = results[0]
                for result in results[1:]:
                    merged.merge_from(result)
                return merged
            return None

    def _initialize_vector_store(self):
        # 1. Create vector store from documents
        try:
            print(f"Initializing vector store from {self.docs_path}")
            logger.info(f"Initializing vector store from {self.docs_path}")
            logger.info(f"What's going on?")
            # 2. Load documents
            loader = DirectoryLoader(self.docs_path, glob="**/*.txt", loader_cls=UnstructuredFileLoader)
            documents = loader.load()
            #documents = documents[:10]
            # 3. Split documents into chunks
            #text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            #splits = text_splitter.split_documents(documents)
            
            logger.info(f"Length of documents", len(documents))
            logger.info(f"Past documents")
            section_documents = []

            for doc in documents:
                raw_text = doc.page_content
                source = doc.metadata.get("source")
                logger.info(f"before split")
                sections = split_by_sections(raw_text)
                logger.info(f"after split")
                for section_name, section_text in sections:
                    if not section_text.strip():
                        continue

                    # Drop junk sections
                    if section_name in DROP_SECTIONS:
                        continue
                    logger.info(f"before split for section")
                    splitter = get_splitter_for_section(section_name)
                    logger.info(f"after split for section")
                    logger.info(section_text)
                    chunks = splitter.split_text(section_text)
                    logger.info(f"after split_text")
                    
                    for chunk in chunks:
                        #logger.info(f"page_content", f"Title: {source}\nSection: {section_name}\n\n{chunk}")
                        section_documents.append(
                            Document(
                                page_content=f"Title: {source}\nSection: {section_name}\n\n{chunk}",
                                metadata={
                                    "source": source,
                                    "section": section_name
                                }
                            )
                        )
            
            
            
            unique_splits = self._deduplicate_chunks(section_documents)

            print(f"Created {len(unique_splits)} unique chunks")
            
            # 4. Initialize embeddings with proper tokenizer configuration
            embeddings = HuggingFaceEmbeddings(
                model_name='sentence-transformers/all-distilroberta-v1', 
                model_kwargs={'device': self.device},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Ensure tokenizer has pad_token set
            if hasattr(embeddings, 'client') and hasattr(embeddings.client, 'tokenizer'):
                if embeddings.client.tokenizer.pad_token is None:
                    embeddings.client.tokenizer.pad_token = embeddings.client.tokenizer.eos_token
            
            # 5. Process in batches
            vectorstore = None
            batch_size = self.batch_size

            logger.warning("DEBUG TYPES:")
            logger.warning(f"type(unique_splits): {type(unique_splits)}")
            logger.warning(f"type(unique_splits[0]): {type(unique_splits[0])}")
            logger.warning(f"batch_size: {self.batch_size} ({type(self.batch_size)})")

            
            print(f"Processing {len(unique_splits)} documents in batches of {batch_size}")
            for i in tqdm(range(0, len(unique_splits), batch_size)):
                batch = unique_splits[i:i+batch_size]
                batch_result = self._process_batch(batch, embeddings)
                
                if batch_result:
                    if vectorstore is None:
                        vectorstore = batch_result
                    else:
                        vectorstore.merge_from(batch_result)
                
                # 6. Save intermediate results periodically
                if i > 0 and i % (batch_size * 10) == 0:
                    with open(f"{self.vectors_path}_partial_{i}", "wb") as f:
                        pickle.dump(vectorstore, f)
            
            # 7. Save final vector store
            with open(self.vectors_path, "wb") as f:
                pickle.dump(vectorstore, f)
            
        except Exception as e:
            raise Exception(f"Failed to initialize vector store: {e}")

    def __call__(self, query: str, n: int) -> list[Document]:
        # 1. Retrieve documents based on query
        cache_key = f"{query}_{n}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 2. Perform search
        results = self.vector_store.similarity_search(query, k=n)
        
        # 3. Cache results
        self.cache[cache_key] = results
        
        return results