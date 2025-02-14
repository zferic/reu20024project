import os
import asyncio
import logging
from fastapi import FastAPI
import sys
sys.path.append("../")
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
from src.models.huggingface import HuggingfaceModel, ModelNames
from src.generator import Generator
from retriever import RerankingRetriever, EmbeddingRetriever
from backend.serialization import serialize_context

class GenerateRequest(BaseModel):
    prompt: str
    context: List[str]
class QueryRequest(BaseModel):
    question: str
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
retriever = RerankingRetriever(EmbeddingRetriever(docs_path= "...", vectors_path="...", logger=logger), first_pass_n = 20)
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
        context = retriever(question, n = 5)
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
# Remove streaming endpoint as it's not needed with the generator API
@app.post("/query")
async def get_query_response(request: QueryRequest):
    try:
        answer = await query(request.question)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {"error": "Unable to process the query."}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, ssl_certfile="./cert.pem", ssl_keyfile="./key.pem")