import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from langchain_core.documents import Document
from backend.src.models.abstract import AbstractModel
from backend.src.models.huggingface import HuggingfaceModel, ModelNames
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Generator:
    def __init__(self, model: AbstractModel):
        # 1. Initialize generator with model
        self.model = model
        self.cache = {}

    def _create_prompt(self, prompt: str, context: list[Document]) -> str:
        # 1. Create formatted prompt with or without context
        if len(context) == 0:
            return f"You are a helpful AI assistant. Answer the following question: {prompt}"
        else:
            context_formatted = "\n\n".join([c.page_content for c in context])
            return (
                "You are a helpful AI assistant who answers questions based on provided context. "
                "The context includes titles, introductions, and results of research studies. "
                "Include the titles of the papers and summaries of the studies in your responses. "
                f"Here is the context: {context_formatted}"
                f"Answer the following question: {prompt}"
            )

    def __call__(self, prompt: str, context: list[Document]) -> str:
        # 1. Generate response for given prompt and context
        context_str = "".join([doc.page_content for doc in context])
        cache_key = f"{prompt}_{hash(context_str)}"
        
        # 2. Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 3. Create the full prompt
        full_prompt = self._create_prompt(prompt, context)
        
        # 4. Generate response
        response = self.model(full_prompt)
        
        # 5. Cache the response
        self.cache[cache_key] = response
        
        return response

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    context: List[str]

# Global variables
model = None
generator = None
is_initializing = False
initialization_complete = False

async def initialize_model_async(background_tasks: BackgroundTasks):
    # 1. Initialize model in background
    global model, generator, is_initializing, initialization_complete
    
    if is_initializing or initialization_complete:
        return
    
    is_initializing = True
    logger.info("Starting model initialization")
    
    try:
        # 2. Initialize model with CPU optimizations
        model = HuggingfaceModel(
            model_name=ModelNames.llama3_2_1B.value,
            max_tokens=1024,
            temperature=0.3,
            use_4bit=True
        )
        
        # 3. Initialize generator
        generator = Generator(model)
        
        initialization_complete = True
        logger.info("Model initialization completed successfully")
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
    finally:
        is_initializing = False

@app.on_event("startup")
async def startup_event():
    # 1. Start background initialization when app starts
    background_tasks = BackgroundTasks()
    await initialize_model_async(background_tasks)

@app.get("/status")
async def get_status():
    # 1. Check initialization status
    global initialization_complete, is_initializing
    
    if initialization_complete:
        return {"status": "ready", "message": "Generator is ready to process requests"}
    elif is_initializing:
        return {"status": "initializing", "message": "Generator is initializing. Please wait."}
    else:
        return {"status": "not_started", "message": "Generator initialization has not started yet"}

@app.post("/generate")
async def generate_response(request: GenerateRequest, background_tasks: BackgroundTasks):
    # 1. Handle generation requests
    global generator, initialization_complete, is_initializing
    
    try:
        # 2. Start initialization if needed
        if not initialization_complete:
            if not is_initializing:
                await initialize_model_async(background_tasks)
            return {"response": "System is initializing. Please try again in a few minutes."}
        
        # 3. Convert context to Document objects
        context_docs = [Document(page_content=text) for text in request.context]
        
        # 4. Generate response
        response = generator(request.prompt, context_docs)
        
        return {"response": response}
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)


