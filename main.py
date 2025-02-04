import os
import asyncio
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
import uvicorn

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

def initialize_vector_store():
    try:
        index_path = "./faiss_index"
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-distilroberta-v1')

        if os.path.exists(index_path):
            logger.info("Loading existing FAISS index...")
            vectorstore = FAISS.load_local(index_path, embeddings)
        else:
            logger.info("Creating new FAISS index...")
            loader = DirectoryLoader(r"C:\Users\tiahi\PROTECTRAG\Research-LLM\papers", glob="**/*.txt")
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = text_splitter.split_documents(documents)
            unique_splits = deduplicate_chunks(splits)

            logger.info("Indexing documents...")
            vectorstore = FAISS.from_documents(unique_splits, embeddings)
            vectorstore.save_local(index_path)

        logger.info("FAISS vector store ready.")
        return vectorstore
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        raise Exception(f"Failed to initialize vector store: {e}")

# reranking
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
vectorstore = None

# initialize vectorstore
@app.on_event("startup")
async def startup_event():
    global vectorstore
    vectorstore = await async_initialize_vector_store()

def retrieve_and_rerank(question, top_k_retrieve=20, top_k_rerank=5):
    try:
        global vectorstore
        initial_results = vectorstore.similarity_search_with_score(question, k=top_k_retrieve)
        pairs = [(question, doc.page_content) for doc, _ in initial_results]
        scores = reranker.predict(pairs)
        scored_results = list(zip(initial_results, scores))
        reranked_results = sorted(scored_results, key=lambda x: x[1], reverse=True)[:top_k_rerank]
        return [(doc.page_content, score) for (doc, _), score in reranked_results]
    except Exception as e:
        logger.error(f"Error in retrieve_and_rerank: {e}")
        return []

def query(question):
    retrieved_contexts = retrieve_and_rerank(question)
    context = "\n\n".join([text for text, _ in retrieved_contexts])
    return f"Answer based on context: {context}"

async def query_async(question):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, query, question)

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def get_query_response(request: QueryRequest):
    try:
        answer = await query_async(request.question)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {"error": "Unable to process the query."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, ssl_certfile="./cert.pem", ssl_keyfile="./key.pem")
