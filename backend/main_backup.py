import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
import logging
import time
import json
import openai
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import requests
from typing import List, Optional
from src.retriever import RerankingRetriever, EmbeddingRetriever
from src.serialization.serialization import serialize_context

### **HARDCODED OPENAI API KEY - REMOVE AFTER DEMO**
openai.api_key = "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"

class QueryRequest(BaseModel):
    question: str

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    feedback: str
    comment: Optional[str] = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize paths
papers_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "papers")
vectors_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectors")
feedback_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback")

# Create feedback directory if it doesn't exist
os.makedirs(feedback_path, exist_ok=True)

# Global variables
retriever = None
is_initializing = False
initialization_complete = False

# Configure thread pool
cpu_count = os.cpu_count() or 4
worker_threads = max(2, cpu_count // 2)
logger.info(f"Configuring thread pool with {worker_threads} workers")
executor = ThreadPoolExecutor(max_workers=worker_threads)

# **FIX CORS**
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://prollm.ece.neu.edu", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def initialize_retriever_async():
    global retriever, is_initializing, initialization_complete
    if is_initializing or initialization_complete:
        return
    is_initializing = True
    logger.info("Starting retriever initialization")
    try:
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
        logger.info("Retriever initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing retriever: {e}")
    finally:
        is_initializing = False

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(initialize_retriever_async())

async def query_openai(question: str):
    global retriever, initialization_complete
    try:
        if not initialization_complete:
            return "System is still initializing. Please try again in a few minutes."

        # Retrieve context from retriever
        context = retriever(question, n=10)
        serialized = serialize_context(context)
        logger.info(f"Context retrieved: {serialized}")

        # Format the prompt based on context
        if not serialized:
            system_message = (
                 "You are a helpful AI assistant who answers questions based only on provided context. "
                "The context includes titles, introductions, and results of research studies. "
                "Include the titles of the papers and summaries of the studies in your responses."
                "Do not use papers outside of PROTECT"
            )
            user_message = f"Answer the following question: {question}"
        else:
            system_message = (
                "You are a helpful AI assistant who answers questions based on provided context. "
                "The context includes titles, introductions, and results of research studies. "
                "When listing papers, always try to include at least 3 relevant papers if available. "
                "Include the titles of the papers and provide detailed summaries of the studies in your responses. "
                "If fewer than 3 papers are available, explain that these are the only relevant papers found in the context. "
            )
            user_message = f"Here is the context:\n{serialized}\n\nAnswer the following question: {question}"

        # Call OpenAI
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature = 0.2
        )

        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in OpenAI query: {e}")
        return f"Error: {str(e)}"

@app.post("/query")
async def get_query_response(request: QueryRequest, background_tasks: BackgroundTasks):
    global initialization_complete
    try:
        if not initialization_complete and not is_initializing:
            background_tasks.add_task(initialize_retriever_async)
            return {"answer": "System is initializing. Please try again shortly."}

        answer = await query_openai(request.question)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {"error": str(e)}

@app.post("/feedback")
async def store_feedback(request: FeedbackRequest):
    try:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        feedback_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "question": request.question,
            "answer": request.answer,
            "feedback": request.feedback,
            "comment": request.comment
        }
        
        feedback_file = os.path.join(feedback_path, f"feedback_{timestamp}.json")
        with open(feedback_file, 'w') as f:
            json.dump(feedback_data, f, indent=2)
        
        logger.info(f"Feedback stored successfully: {feedback_file}")
        return {"status": "success", "message": "Feedback stored successfully"}
    except Exception as e:
        logger.error(f"Error storing feedback: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, ssl_certfile="./cert.pem", ssl_keyfile="./key.pem")
