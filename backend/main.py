import os
import asyncio
import logging
from fastapi import FastAPI
import sys

# Add all possible paths to handle both old and new structures
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "src"))

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
from typing import List

# Try importing from new structure, fall back to old if needed
try:
    from backend.src.models.huggingface import HuggingfaceModel, ModelNames
    from backend.src.generator import Generator
    from backend.src.retriever import RerankingRetriever, EmbeddingRetriever
    from backend.src.serialization.serialization import serialize_context
    print("Using new module structure")
except ImportError:
    try:
        from src.models.huggingface import HuggingfaceModel, ModelNames
        from src.generator import Generator
        from src.retriever import RerankingRetriever, EmbeddingRetriever
        from src.serialization.serialization import serialize_context
        print("Using src module structure")
    except ImportError:
        from retriever import RerankingRetriever, EmbeddingRetriever
        from models.huggingface import HuggingfaceModel, ModelNames
        from generator import Generator
        # Define a simple serialization function if the module is not found
        def serialize_context(documents):
            return [doc.page_content for doc in documents]
        print("Using old module structure")

class GenerateRequest(BaseModel):
    prompt: str
    context: List[str]

class QueryRequest(BaseModel):
    question: str

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize retriever with proper paths
papers_path = os.path.join(project_root, "papers")
if not os.path.exists(papers_path):
    # Try alternative paths
    papers_path = os.path.join(current_dir, "papers")
    if not os.path.exists(papers_path):
        papers_path = "papers"  # Fallback to relative path

vectors_path = os.path.join(current_dir, "vectors")
if not os.path.exists(os.path.dirname(vectors_path)):
    vectors_path = "vectors"  # Fallback to relative path

print(f"Papers path: {papers_path}")
print(f"Vectors path: {vectors_path}")

try:
    retriever = RerankingRetriever(
        EmbeddingRetriever(
            docs_path=papers_path,
            vectors_path=vectors_path,
            recreate=False  
        ),
        first_pass_n=20
    )
except Exception as e:
    logger.error(f"Error initializing retriever: {e}")
    raise

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor()

async def query(question: str):
    try:
        context = retriever(question, n=5)
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
        return "Error: Unable to complete the query."

@app.post("/query")
async def get_query_response(request: QueryRequest):
    try:
        answer = await query(request.question)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {"error": "Unable to process the query."}

if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, ssl_certfile="./cert.pem", ssl_keyfile="./key.pem")
    except FileNotFoundError:
        logger.warning("SSL certificates not found, running without HTTPS")
        uvicorn.run(app, host="0.0.0.0", port=8000)