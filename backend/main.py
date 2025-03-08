import os
import asyncio
import logging
import time
import multiprocessing
from fastapi import FastAPI, BackgroundTasks
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
import uvicorn
import requests
from fastapi.responses import StreamingResponse
from typing import List, Optional
from backend.src.models.huggingface import HuggingfaceModel, ModelNames
from backend.src.generator import Generator
from backend.src.retriever import RerankingRetriever, EmbeddingRetriever
from backend.src.serialization.serialization import serialize_context

class GenerateRequest(BaseModel):
    prompt: str
    context: List[str]

class QueryRequest(BaseModel):
    question: str

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize paths
papers_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "papers")
vectors_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectors")

# Global variables
retriever = None
is_initializing = False
initialization_complete = False

# Configure thread pool based on CPU count
cpu_count = os.cpu_count() or 4
# Use fewer threads to avoid overloading the CPU
worker_threads = max(2, cpu_count // 2)
logger.info(f"Configuring thread pool with {worker_threads} workers")
executor = ThreadPoolExecutor(max_workers=worker_threads)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def initialize_retriever_async():
    """Initialize the retriever in the background"""
    global retriever, is_initializing, initialization_complete
    
    if is_initializing or initialization_complete:
        return
    
    is_initializing = True
    logger.info("Starting background initialization of retriever")
    
    try:
        # Run initialization in a separate thread to not block the main thread
        loop = asyncio.get_event_loop()
        retriever = await loop.run_in_executor(
            executor,
            lambda: RerankingRetriever(
                EmbeddingRetriever(
                    docs_path=papers_path,
                    vectors_path=vectors_path,
                    recreate=False,
                    batch_size=32  # Adjust batch size based on available memory
                ),
                first_pass_n=20,
                batch_size=8  # Smaller batch size for reranking to reduce memory usage
            )
        )
        
        initialization_complete = True
        logger.info("Retriever initialization completed successfully")
    except Exception as e:
        logger.error(f"Error initializing retriever: {e}")
    finally:
        is_initializing = False

@app.on_event("startup")
async def startup_event():
    """Start background initialization when the app starts"""
    asyncio.create_task(initialize_retriever_async())

async def query(question: str):
    """Process a query using the retriever and generator"""
    global retriever, initialization_complete
    
    try:
        # Check if retriever is initialized
        if not initialization_complete:
            logger.info("Retriever not yet initialized, waiting...")
            # Wait for initialization to complete with a timeout
            start_time = time.time()
            timeout = 300  # 5 minutes timeout
            
            while not initialization_complete and time.time() - start_time < timeout:
                await asyncio.sleep(1)
            
            if not initialization_complete:
                return "System is still initializing. Please try again in a few minutes."
        
        # Retrieve context
        start_time = time.time()
        context = retriever(question, n=5)
        logger.info(f"Context retrieval completed in {time.time() - start_time:.2f} seconds")
        
        # Serialize context
        serialized = serialize_context(context)
        
        # Make request to generator API
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            executor,
            lambda: requests.post(
                "http://localhost:8001/generate",
                json={
                    "prompt": question,
                    "context": serialized
                }
            )
        )
        
        response_data = response.json()
        return response_data.get("response", "No response received.")
    except Exception as e:
        logger.error(f"Error in query: {e}")
        return f"Error: Unable to complete the query. {str(e)}"

@app.get("/status")
async def get_status():
    """Check the initialization status of the system"""
    global initialization_complete, is_initializing
    
    if initialization_complete:
        return {"status": "ready", "message": "System is ready to process queries"}
    elif is_initializing:
        return {"status": "initializing", "message": "System is initializing. Please wait."}
    else:
        return {"status": "not_started", "message": "System initialization has not started yet"}

@app.post("/query")
async def get_query_response(request: QueryRequest, background_tasks: BackgroundTasks):
    """Handle query requests"""
    global initialization_complete
    
    try:
        # If not initialized, start initialization
        if not initialization_complete and not is_initializing:
            background_tasks.add_task(initialize_retriever_async)
            return {"answer": "System is initializing. Please try again in a few minutes."}
        
        answer = await query(request.question)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {"error": f"Unable to process the query: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, ssl_certfile="./cert.pem", ssl_keyfile="./key.pem")