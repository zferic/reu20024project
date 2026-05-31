import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append("/var/www/reu20024project")
from langchain_core.documents import Document
from backend.src.models.abstract import AbstractModel
from backend.src.models.huggingface import HuggingfaceModel, ModelNames
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import uvicorn
from backend.src.models.huggingface import LlamaCppModel
from backend.src.models.hf_inference_provider import HFInferenceProviderModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Generator:
    def __init__(self, model: AbstractModel):
        # 1. Initialize generator with model
        self.model = model
        self.cache = {}

    def wrap_llama2_chat(self, prompt):
        return f"<s>[INST] <<SYS>>\nYou are a helpful AI assistant.\n<</SYS>>\n\n{prompt} [/INST]"

    def _create_prompt(self, prompt: str, context: list[Document]) -> str:
        # 1. Create formatted prompt with or without context
        if len(context) == 0:
            return f"You are a helpful AI assistant. Answer the following question: {prompt}"
        else:
            context_formatted = "\n\n".join([c.page_content for c in context])
            return (
                "You are a helpful AI assistant who answers using the provided context.\n"
                "Rules:\n"
          
                f"Context:\n{context_formatted}\n\n"
                f"Question: {prompt}\n"

                )

    def __call__(self, prompt: str, context: list[Document]) -> str:
        # 1. Generate response for given prompt and context
        context_str = "".join([doc.page_content for doc in context])
        cache_key = f"{prompt}_{hash(context_str)}"
        
        # 2. Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 3. Create the full prompt
        full_prompt = self.wrap_llama2_chat(self._create_prompt(prompt, context))
        #full_prompt = self._create_prompt(prompt, context)

        logger.info(f"Full Prompt: {full_prompt}")
        
        # 4. Generate response
        response = self.model(full_prompt)
        
        # 5. Cache the response
        self.cache[cache_key] = response
        
        return response

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    context: List[str]

APP_VERSION = "hf-provider-test-2026-01-22"
MODEL_BACKEND = "hf_provider"
#MODEL_BACKEND = os.getenv("MODEL_BACKEND", "llamacpp")  # "llamacpp" or "hf_provider"
HF_PROVIDER_MODEL = os.getenv("HF_PROVIDER_MODEL", "openai/gpt-oss-120b")

# Global variables
model = None
generator = None
is_initializing = False
initialization_complete = False

async def initialize_model_async(background_tasks: BackgroundTasks):
    # 1. Initialize model in background
    global model, generator, is_initializing, initialization_complete

    print("Available models:")
    for m in ModelNames:
        print("-", m.name, "→", m.value)

    if is_initializing or initialization_complete:
        return

    is_initializing = True
    logger.info("Starting model initialization")

    try:
        if MODEL_BACKEND == "hf_provider":
            '''
            model = HuggingfaceModel(
                model_name=ModelNames.llama3_2_3B.value,
                max_tokens=256,
                temperature=0.3,
                use_4bit=True
            )
            '''
            model = HFInferenceProviderModel(
                model_name=HF_PROVIDER_MODEL,
                max_tokens=10000,
                temperature=0.3,
            )
            logger.info(f"Using HF provider model: {HF_PROVIDER_MODEL}")
        else:
            model = LlamaCppModel(
                model_path="/home/zlatan7369/reu20024project/backend/src/generator/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                max_tokens=512,
                temperature=0.3,
                n_threads=6,
            )
            logger.info("Using local LlamaCpp model")

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
    global initialization_complete, is_initializing

    if initialization_complete:
        return {"status": "ready", "message": "Generator is ready to process requests", "version": APP_VERSION}
    elif is_initializing:
        return {"status": "initializing", "message": "Generator is initializing. Please wait.", "version": APP_VERSION}
    else:
        return {"status": "not_started", "message": "Generator initialization has not started yet", "version": APP_VERSION}


@app.post("/generate_stream")
async def generate_stream(request: GenerateRequest):
    global generator, initialization_complete

    if not initialization_complete:
        return StreamingResponse(iter(["System initializing..."]), media_type="text/plain")

    context_docs = [Document(page_content=text) for text in request.context]
    prompt = generator._create_prompt(request.prompt, context_docs)
    full_prompt = generator.wrap_llama2_chat(prompt)

    def token_stream():
        # HF provider path (our wrapper exposes .stream())
        if hasattr(generator.model, "stream"):
            for chunk in generator.model.stream(full_prompt):
                yield chunk
            return

        # llama.cpp fallback (your existing logic)
        for event in generator.model.model(
            prompt=full_prompt,
            max_tokens=generator.model.max_tokens,
            temperature=generator.model.temperature,
            stream=True,
        ):
            if "choices" in event:
                text = event["choices"][0].get("text", "")
                if text:
                    yield text

    return StreamingResponse(token_stream(), media_type="text/plain")

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


