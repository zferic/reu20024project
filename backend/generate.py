import sys
from langchain_core.documents import Document
sys.path.append("./")
from src.generator import Generator
from src.models.abstract import AbstractModel
from src.models.huggingface import HuggingfaceModel, ModelNames
from backend.src.serialization.serialization import deserialize_context
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    context: List[dict]

model = HuggingfaceModel(
    model_name=ModelNames.llama3_2_1B.value,
    max_tokens=1024,
    temperature=0.3
)
generator = Generator(model)

@app.post("/generate")
async def generate_response(request: GenerateRequest):
    try:
        context_docs = deserialize_context(request.context)
        response = generator(request.prompt, context_docs)
        
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)


