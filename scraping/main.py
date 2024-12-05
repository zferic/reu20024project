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
import torch
import faiss
import pickle

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
DATABASE_URL = "DATABASE_URL"

engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

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

embeddings_file = r'C:\Users\tiahi\PROTECTRAG\Research-LLM\embeddings.pkl'
if os.path.exists(embeddings_file):
    with open(embeddings_file, 'rb') as f:
        data = pickle.load(f)
    embeddings = data['embeddings']
    contexts = data['contexts']
    faiss.normalize_L2(embeddings)
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(embeddings)
else:
    raise FileNotFoundError("Embeddings file not found. Please ensure embeddings.pkl exists.")

def initialize_model_and_tokenizer(model_name='sentence-transformers/all-distilroberta-v1'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    return tokenizer, model, device

def get_query_embedding(question, tokenizer, model, device):
   
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    question_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
    return question_embedding

def retrieve_closest_texts(question, top_n=5):

    tokenizer, model, device = initialize_model_and_tokenizer()
    question_embedding = get_query_embedding(question, tokenizer, model, device).astype('float32')
    faiss.normalize_L2(question_embedding.reshape(1, -1))
    
    distances, indices = faiss_index.search(question_embedding.reshape(1, -1), top_n)
    
    closest_texts = [contexts[i] for i in indices[0]]
    
    closest_texts_with_distances = [
        (closest_texts[i], distances[0][i]) for i in range(len(closest_texts))
    ]
    
    return closest_texts_with_distances


api_key = 'OPENAI_API_KEY'
client = OpenAI(api_key=api_key)

def query(question):
    top_contexts = retrieve_closest_texts(question, top_n=5)
    context = " | ".join([f"Concept {i + 1} - {text}" for i, text in enumerate(top_contexts)])
    system_msg = (  
        """You are a helpful AI assistant for answering questions based on the context provided. \
        The context I provide includes the title, introduction, and results of research studies. \
        Include the titles of the paper and summary of the study in responses."""
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Context: {context}\\n\\nQuestion: {question}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return "Error: Unable to process the query."

async def query_async(question):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, query, question)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

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
