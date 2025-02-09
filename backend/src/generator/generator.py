import sys
from langchain_core.documents import Document
sys.path.append("./")
from src.models.abstract import AbstractModel
from src.models.huggingface import HuggingfaceModel, ModelNames
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

class Generator:

    def __init__(self, model : AbstractModel):
        self.model = model

    def __call__(self, prompt : str, context : list[Document]) -> str:
        if len(context) == 0:
            full_prompt = f"You are a helpful AI assistant. Answer the following question: {prompt}"
        else:
            context_formatted = "\n\n".join([c.page_content for c in context])
            full_prompt = (
            "You are a helpful AI assistant who answers questions based on provided context. "
            "The context includes titles, introductions, and results of research studies. "
            "Include the titles of the papers and summaries of the studies in your responses. "
            f"Here is the context: {context_formatted}"
            f"Answer the following question: {prompt}"
            )
        return self.model(full_prompt)

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    context: List[str]

model = HuggingfaceModel(
    model_name=ModelNames.llama3_2_1B.value,
    max_tokens=1024,
    temperature=0.3
)
generator = Generator(model)

@app.post("/generate")
async def generate_response(request: GenerateRequest):
    try:
        context_docs = [Document(page_content=text) for text in request.context]
        response = generator(request.prompt, context_docs)
        
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)


