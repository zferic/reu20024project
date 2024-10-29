import os
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import textwrap
import time
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/allenai-specter', clean_up_tokenization_spaces=True)
model = AutoModel.from_pretrained('sentence-transformers/allenai-specter')

api_key = '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'
client = OpenAI(api_key=api_key)


embeddings_df = pd.read_csv(r"C:\Users\tiahi\PROTECTRAG\Research-LLM\embeddings_prime.csv")
complete_texts_df = pd.read_csv(r"C:\Users\tiahi\PROTECTRAG\Research-LLM\final_texts.csv")

columns_to_process = ["title", "authors", "publication_date", "abstract", "introduction", "results", "discussion", "conclusion"]
def chunk_text(text, max_tokens=8192):
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunks.append(" ".join(words[start:end]))
        start = end
    
    return chunks


def query(question, max_context_length=4096):
    question_chunks = chunk_text(question, max_tokens=8192)  

    question_embeddings = []
    for chunk in question_chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=512)
        question_embedding = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy().flatten()
        question_embeddings.append(question_embedding)
        question_embedding = np.mean(question_embeddings, axis=0)

    def fn(question_embedding, page_embedding):
        if page_embedding is None or isinstance(page_embedding, float):  
            return -np.inf
        page_embedding = np.array(eval(page_embedding), dtype=np.float32)
    
   
        return np.dot(page_embedding, question_embedding)

    distances = []
    for col in columns_to_process:
        col_distance_series = embeddings_df[col].apply(lambda x: fn(question_embedding, x))
        distances.append(col_distance_series)


    combined_distances = sum(distances)
    combined_distances.sort_values(ascending=False, inplace=True)

    top_four_indices = combined_distances.index[:4]

    contexts = []
    for col in columns_to_process:
        text_series = complete_texts_df.loc[top_four_indices, col]
        contexts.extend(text_series.dropna().astype(str).tolist())

    context = "\n\n".join(contexts)

    if len(context) > max_context_length:
        context = context[:max_context_length] + "..."

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are meant to answer queries with PROTECT initiative research from the data I have provided for you. Here is the context:\n\n{context}"},
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error: Unable to complete the query due to quota limitations."
while(True):
    question = input("Enter a query:")
    print(query(question))
    
