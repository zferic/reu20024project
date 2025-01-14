import os
import pandas as pd
import numpy as np
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import requests

from langchain.docstore.document import Document as LangchainDocument
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredFileLoader
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
executor = ThreadPoolExecutor()
def deduplicate_chunks(chunks):
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
async def async_initialize_vector_store():
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, initialize_vector_store)

import re
# Function to extract sections from a paper
def extract_sections(text):
    # Match the Title
    title_match = re.search(r'Title:\s*(.+?)\n', text)

    # Match the Authors
    authors_match = re.search(r'Authors:\s*(.+?)\n', text)

    # Match the Publication Date
    publication_date_match = re.search(r'Publication Date:\s*(.+?)\n', text)

    abstract_match = re.search(r'### Abstract ###(.*?)### Introduction ###', text, re.DOTALL)
    introduction_match = re.search(r'### Introduction ###(.*?)(###|$)', text, re.DOTALL)
    methods_match = re.search(r'### Methods ###(.*?)(###|$)', text, re.DOTALL)
    results_match = re.search(r'### Results ###(.*?)(###|$)', text, re.DOTALL)
    conclusion_match = re.search(r'### Conclusion ###(.*?)(###|$)', text, re.DOTALL)
    

    # Extract the groups if matches are found
    title = title_match.group(1) if title_match else ""
    authors = authors_match.group(1) if authors_match else ""
    publication_date = publication_date_match.group(1) if publication_date_match else ""
    abstract = abstract_match.group(1).strip() if abstract_match else ""
    introduction = introduction_match.group(1).strip() if introduction_match else ""
    methods = methods_match.group(1).strip() if methods_match else ""
    results = results_match.group(1).strip() if results_match else ""
    conclusion = conclusion_match.group(1).strip() if conclusion_match else ""
    
    return abstract, introduction, methods, results, conclusion, title, authors, publication_date

def load_documents():
    # Define the directory containing the text files
    text_files_dir = "/media/zman/extrahd/reu20024project/preprocessing/docs"

    # Initialize lists to store the extracted sections
    abstracts = []
    introductions = []
    methods = []
    results = []
    conclusions = []

    # Read all text files and extract sections
    for filename in os.listdir(text_files_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(text_files_dir, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                abstract, introduction, method, result, conclusion, title, authors, publication_date = extract_sections(text)
                source_doc = filename.replace("_sectionts.xtx", "")

                metadata = {
                'title': title,
                'authors': authors,
                'publication_date': publication_date,
                'source': source_doc
                }

                print(metadata)

                if len(abstract.strip()) > 100:
                    abstracts.append({'text': abstract.replace('\n', ' ').strip(), **metadata, 'section': 'Abstract'})
                
                if len(introduction.strip()) > 100:
                    introductions.append({'text': introduction.replace('\n', ' ').strip(), **metadata, 'section': 'Introduction'})
                
                if len(method.strip()) > 100:
                    methods.append({'text': method.replace('\n', ' ').strip(), **metadata, 'section': 'Methods'})
                
                if len(result.strip()) > 100:
                    results.append({'text': result.replace('\n', ' ').strip(), **metadata, 'section': 'Results'})
                
                if len(conclusion.strip()) > 100:
                    conclusions.append({'text': conclusion.replace('\n', ' ').strip(), **metadata, 'section': 'Conclusion'})

    
    RAW_KNOWLEDGE_BASE = [
        LangchainDocument(
            page_content=doc["text"],
            metadata={
                "source": doc["source"],
                "title": doc.get("title", "Unknown Title"),
                "authors": doc.get("authors", "Unknown Authors"),
                "publication_date": doc.get("publication_date", "Unknown Date"),
                "section": doc.get("section", "Unknown Section")
            }
        )
        for doc in abstracts + introductions + methods + results + conclusions
    ]

    return RAW_KNOWLEDGE_BASE


def initialize_vector_store():
    """
    Initialize or load a FAISS vector store.
    If the FAISS index exists at the specified path, it will be loaded.
    Otherwise, the index will be created and saved to the path.

    :return: The initialized FAISS vector store.
    """
    index_path = '/media/zman/extrahd/reu20024project/faiss_index_55555'  # Hardcoded index path
    
    try:
        # Check if the FAISS index already exists
        if os.path.exists(index_path):
            logger.info(f"Loading existing FAISS index from {index_path}...")
            vectorstore = FAISS.load_local(index_path, 
                                           HuggingFaceEmbeddings(model_name='sentence-transformers/all-distilroberta-v1'),
                                            allow_dangerous_deserialization=True  # Explicitly allow deserialization
            )
            logger.info("FAISS vector store loaded successfully.")
        else:
            logger.info("FAISS index not found. Creating a new index...")
            #loader = DirectoryLoader(r'/media/zman/extrahd/reu20024project/papers', glob="**/*.txt", loader_cls=UnstructuredFileLoader)
           # documents = loader.load()

            MARKDOWN_SEPARATORS = [
                "\n#{1,6} ",
                "```\n",
                "\n\\*\\*\\*+\n",
                "\n---+\n",
                "\n___+\n",
                "\n\n",
                "\n",
                " ",
                "",
            ]

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # The maximum number of characters in a chunk: we selected this value arbitrarily
                chunk_overlap=100,  # The number of characters to overlap between chunks
                add_start_index=True,  # If `True`, includes chunk's start index in metadata
                strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
                separators=MARKDOWN_SEPARATORS,
            )

            RAW_KNOWLEDGE_BASE = load_documents()

            docs_processed = []
            for doc in RAW_KNOWLEDGE_BASE:
                docs_processed += text_splitter.split_documents([doc])
            #unique_splits = deduplicate_chunks(splits)
            logger.info("Creating FAISS index...")
            EMBEDDING_MODEL_NAME ='sentence-transformers/all-distilroberta-v1'

            embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                multi_process=True,
                model_kwargs={"device": "cuda"},
                encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
            )

            #vectorstore = FAISS.from_documents(unique_splits, embeddings)

            vectorstore = FAISS.from_documents(
            docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE)
            vectorstore.save_local(index_path)
            logger.info(f"FAISS vector store created and saved to {index_path} successfully.")
        
        return vectorstore
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        raise Exception(f"Failed to initialize vector store: {e}")
    
from ragatouille import RAGPretrainedModel
reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

#reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
vectorstore = None
@app.on_event("startup")
async def startup_event():
    global vectorstore
    vectorstore = await async_initialize_vector_store()


def retrieve_and_rerank2(question, top_k_retrieve=20, top_k_rerank=4):
    global vectorstore
    relevant_docs = vectorstore.similarity_search(
        query=question, k=top_k_retrieve
    )

     # Keep both content and metadata
    relevant_docs_with_metadata = [
        {"content": doc.page_content, "metadata": doc.metadata} 
        for doc in relevant_docs
    ]

    # Extract only content for reranker
    relevant_contents = [doc["content"] for doc in relevant_docs_with_metadata]

    reranked_contents = reranker.rerank(question, relevant_contents, k=top_k_rerank)

    reranked_indices = [relevant_contents.index(doc["content"]) for doc in reranked_contents]

        # Reattach metadata based on reranked indices
    relevant_docs_with_metadata = [
            relevant_docs_with_metadata[i] for i in reranked_indices
        ]

    relevant_docs_with_metadata = relevant_docs_with_metadata[:top_k_rerank]

    context = "\nExtracted documents:\n"
    include_metadata = True

    print("Printing doc relevant with metadata" , relevant_docs_with_metadata[0])

    if include_metadata:
        # Include metadata in the context
        context += "".join(
            [
                f"Publication: {doc['metadata'].get('source', 'Unknown')}\n"
                f"Title: {doc['metadata'].get('title', 'Unknown Title')}\n"
                f"Authors: {doc['metadata'].get('authors', 'Unknown Authors')}\n"
                f"Publication Date: {doc['metadata'].get('publication_date', 'Unknown Date')}\n\n"
                + doc["content"] + "\n"
                for i, doc in enumerate(relevant_docs_with_metadata)
            ]
        )
    else:
        # Use only the content
        context += "".join(
            [
                f"Document {str(i)}:::\n"
                + doc["content"] + "\n"
                for i, doc in enumerate(relevant_docs_with_metadata)
            ]
        )

    return context

    
def retrieve_and_rerank(question, top_k_retrieve=20, top_k_rerank=7):
    try:
        global vectorstore
        
        initial_results = vectorstore.similarity_search_with_score(question, k=top_k_retrieve)

        pairs = [(question, doc.page_content) for doc, _ in initial_results]

        scores = reranker.predict(pairs)
        scored_results = list(zip(initial_results, scores))
        reranked_results = sorted(scored_results, key=lambda x: x[1], reverse=True)[:top_k_rerank]
        return [(doc.page_content, score) for (doc, _), score in reranked_results]


    except Exception as e:
        print(f"Error in retrieve_and_rerank: {e}")
        return []

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")


def query(question):

    

    contexts = retrieve_and_rerank2(question)
    #contexts = "\n\n".join([f"Doc {i+1}: {text.strip()}" for i, (text, _) in enumerate(retrieved_contexts)])


    prompt_in_chat_format = [
    {
        "role": "system",
        "content":  """You are a helpful environmental assistant. Your job is to answer the user's question based only on the provided context. 
        "Do not ask any additional questions. Always provide a clear and concise answer. 
        "Include the titles and dates the papers in your response.\n\n"""
    },
    {
        "role": "user",
        "content": """Context:
        {contexts}
        ---
        Now here is the question you need to answer.

        Question: {question}""",
            },
    ]
        
    RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
        prompt_in_chat_format, tokenize=False, add_generation_prompt=True
    )
    print(contexts)
    print(RAG_PROMPT_TEMPLATE)

    final_prompt = RAG_PROMPT_TEMPLATE.format(
                question=question, contexts=contexts
    )
    prompt = (
        "You are a helpful environmental assistant. Your job is to answer the user's question based only on the provided context. "
        "Do not ask any additional questions. Always provide a clear and concise answer. "
        "When applicable, include the titles of the referenced papers in your response.\n\n"
        f"Context:\n{contexts}\n\nQuestion: {question}\nAnswer:"
    )


    #print("Context:", contexts)
    print("Prompt:", final_prompt)
    try:
        response = requests.post("http://localhost:8001/generateb", json={"prompt": final_prompt, "max_tokens": 1024})
        print("response:", response)
        response.raise_for_status()
        response_data = response.json()
        print(response_data)
        return response_data.get("response", "No response received.")
    except Exception as e:
        print(f"Error: {e}")
        return "Error: Unable to process the query."
    
async def query_async(question):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, query, question)
class QueryRequest(BaseModel):
    question: str
@app.post("/query")
async def get_query_response(request: QueryRequest):
    answer = await query_async(request.question)
    return {"answer": answer}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)