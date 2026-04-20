import os
import asyncio
import logging
import time
from fastapi import FastAPI, BackgroundTasks
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
from langchain_core.documents import Document
import uvicorn
import requests
from typing import List
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
logger.info("GENERATOR_URL: " + str(os.getenv("GENERATOR_URL")))

# Initialize paths
papers_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "papers")
vectors_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectors")

# Global variables
retriever = None
is_initializing = False
initialization_complete = False

# Configure thread pool
cpu_count = os.cpu_count() or 4
worker_threads = max(2, cpu_count // 2)
logger.info(f"Configuring thread pool with {worker_threads} workers")
executor = ThreadPoolExecutor(max_workers=worker_threads)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def initialize_retriever_async():
    # 1. Initialize retriever in background
    global retriever, is_initializing, initialization_complete
    
    if is_initializing or initialization_complete:
        return
    
    is_initializing = True
    logger.info("Starting background initialization of retriever")
    
    try:
        # 2. Run initialization in a separate thread
        loop = asyncio.get_event_loop()
        retriever = await loop.run_in_executor(
            executor,
            lambda: RerankingRetriever(
                EmbeddingRetriever(
                    docs_path=papers_path,
                    vectors_path=vectors_path,
                    recreate=False,
                    batch_size=32
                ),
                first_pass_n=20,
                batch_size=8
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
    # 1. Start background initialization when app starts
    asyncio.create_task(initialize_retriever_async())

async def query(question: str):
    global retriever, initialization_complete
    try:
        if not initialization_complete:
            logger.info("Retriever not yet initialized, waiting...")
            start_time = time.time()
            timeout = 300  
            while not initialization_complete and time.time() - start_time < timeout:
                await asyncio.sleep(1)
            if not initialization_complete:
                return "System is still initializing. Please try again in a few minutes."
        
        context = retriever(question, n=3)
        serialized = serialize_context(context)

        serialized = [doc.page_content for doc in context]

        print("Serialized context:")
        print(serialized)

        logger.info(f"Sending to generator: prompt={question}, context={serialized}")
        
        loop = asyncio.get_event_loop()

        gen_url = os.getenv("GENERATOR_URL", "http://127.0.0.1:8007/generate")
        logger.info(f"GENERATOR_URL resolved to: {gen_url}")
        response = await loop.run_in_executor(
            executor,
            lambda: requests.post(
                "http://127.0.0.1:8007/generate",  # pick the real port
                json={"prompt": question, "context": serialized},  # ok to test with empty context
                headers={"accept": "application/json"},
                timeout=1000
            )
        )

        if response.status_code != 200:
            logger.error(
                f"Generator HTTP {response.status_code} | "
                f"CT={response.headers.get('content-type')} | "
                f"Body={response.text[:500]}"
            )
            return f"Generator error ({response.status_code}): {response.text[:200]}"

        try:
            response_data = response.json()
        except Exception:
            logger.error(
                f"Non-JSON from generator | "
                f"CT={response.headers.get('content-type')} | "
                f"Body={response.text[:500]}"
            )
            return "Generator returned a non-JSON response."

        try:
            response_data = response.json()
            print("Printing response data:")
            print(response_data)
            logger.info(f"Generator response raw: {response.text}")
            logger.info(f"Generator response JSON: {response_data}")

        except Exception as parse_err:
            logger.error(f"Failed to parse JSON response: {parse_err}")
            print(f"Failed to parse JSON response: {parse_err}")
            return "No response received."

        logger.info(f"Generator response: {response_data}")
        return response_data.get("response", "No response received.")
    except Exception as e:
        logger.error(f"Error in query: {e}")
        return f"Error: Unable to complete the query. {str(e)}"

@app.get("/status")
async def get_status():
    # 1. Check initialization status
    global initialization_complete, is_initializing
    
    if initialization_complete:
        return {"status": "ready", "message": "System is ready to process queries"}
    elif is_initializing:
        return {"status": "initializing", "message": "System is initializing. Please wait."}
    else:
        return {"status": "not_started", "message": "System initialization has not started yet"}

@app.post("/api/query")
async def get_query_response(request: QueryRequest, background_tasks: BackgroundTasks):
    # 1. Handle query requests
    global initialization_complete
    
    try:
        # 2. Start initialization if needed
        if not initialization_complete and not is_initializing:
            background_tasks.add_task(initialize_retriever_async)
            return {"answer": "System is initializing. Please try again in a few minutes."}
        
        # 3. Process query and return answer
        answer = await query(request.question)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {"error": f"Unable to process the query: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)#, ssl_certfile="./cert.pem", ssl_keyfile="./key.pem")