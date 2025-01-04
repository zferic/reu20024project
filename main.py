import os
import pandas as pd
import numpy as np
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, Text, DateTime, func
import asyncio
from concurrent.futures import ThreadPoolExecutor
import requests
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredFileLoader
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
DATABASE_URL = "DATABASE_URL"

engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserQuestion(Base):
    __tablename__ = 'user_questions'
    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

class ModelResponse(Base):
    __tablename__ = 'model_responses'
    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer)
    response = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

class UserFeedback(Base):
    __tablename__ = 'user_feedbacks'
    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer)
    response_id = Column(Integer)
    feedback = Column(Text)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

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
        loader = DirectoryLoader(r'C:\Users\tiahi\PROTECTRAG\Research-LLM\papers', glob="**/*.txt", loader_cls=UnstructuredFileLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(documents)

        unique_splits = deduplicate_chunks(splits)
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-distilroberta-v1')
        vectorstore = FAISS.from_documents(unique_splits, embeddings)
        return vectorstore
    except Exception as e:
        raise Exception(f"Failed to initialize vector store: {e}")



reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
vectorstore = None

@app.on_event("startup")
async def startup_event():
    global vectorstore
    vectorstore = await async_initialize_vector_store()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


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
        print(f"Error in retrieve_and_rerank: {e}")
        return []

def query(question):
    retrieved_contexts = retrieve_and_rerank(question)
    context = "\n\n".join([text for text, _ in retrieved_contexts])
    prompt = (
       "You are a helpful AI assistant who answers questions based on provided context. "
     "The context includes titles, introductions, and results of research studies. "
     "Include the titles of the papers and summaries of the studies in your responses. "
     f"Answer the following question: {question}"
    )
    try:
        response = requests.post("http://localhost:8001/generate", json={"prompt": prompt, "max_tokens": 512})
        response.raise_for_status()
        response_data = response.json()
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
async def get_query_response(request: QueryRequest, db: AsyncSession = Depends(get_db)):
    new_question = UserQuestion(question=request.question)
    db.add(new_question)
    await db.commit()
    await db.refresh(new_question)

    answer = await query_async(request.question)

    new_response = ModelResponse(question_id=new_question.id, response=answer)
    db.add(new_response)
    await db.commit()
    await db.refresh(new_response)

    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
