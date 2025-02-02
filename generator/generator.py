import sys
from langchain_core.documents import Document
sys.path.append("./")
from models.abstract import AbstractModel

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


