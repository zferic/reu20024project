import os
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import textwrap
import time
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/allenai-specter')
model = AutoModel.from_pretrained('sentence-transformers/allenai-specter')

api_key = 'OPENAI_API_KEY'
client = OpenAI(api_key=api_key)

def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def chunk_text(text, max_tokens=512):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunks.append(" ".join(words[start:end]))
        start = end
    return chunks

def get_embedding_safe(text, model, tokenizer):
    if isinstance(text, str):
        text = text.replace("\n", " ")
        try:
            text_chunks = chunk_text(text, max_tokens=512)
            embeddings = []
            for chunk in text_chunks:
                embedding = get_embedding(chunk, model, tokenizer)
                embeddings.append(embedding)
            return np.mean(embeddings, axis=0)
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    else:
        return None

df = pd.read_csv(r"C:\Users\tiahi\NSF REU\tokenizing\RAG\final_texts.csv")

columns_to_process = ["title", "authors", "publication_date", "abstract", "introduction", "results", "discussion", "conclusion"]

embeddings_dict = {col: [] for col in columns_to_process}

for idx, row in df.iterrows():
    for col in columns_to_process:
        text = row[col]
        embedding = get_embedding_safe(text, model, tokenizer)
        embeddings_dict[col].append(embedding)

embeddings_df = pd.DataFrame(embeddings_dict)
for col in columns_to_process:
    embeddings_df[col] = embeddings_df[col].apply(lambda emb: emb.tolist() if isinstance(emb, np.ndarray) else emb)

embeddings_df.to_csv("embeddings.csv", index=False)
print(f"Saved embeddings to 'embeddings.csv'")

def query(question, max_context_length=4096):
    question_embedding = get_embedding_safe(question, model, tokenizer)
    
    def fn(question_embedding, page_embedding):
        if page_embedding is None:
            return -np.inf
        page_embedding = np.array(page_embedding, dtype=np.float32)
        if page_embedding.size == 0:
            return -np.inf
        return np.dot(page_embedding, question_embedding)
    
    distances = []
    for col in columns_to_process:
        col_distance_series = embeddings_df.apply(lambda row: fn(question_embedding, row[col]), axis=1)
        distances.append(col_distance_series)
    
    combined_distances = sum(distances)
    combined_distances.sort_values(ascending=False, inplace=True)
    
    top_four_indices = combined_distances.index[:4]

    contexts = []
    for col in columns_to_process:
        text_series = df.loc[top_four_indices, col]
        contexts.extend(text_series.dropna().astype(str).tolist()) 

    context = "\n\n".join(contexts)

    encoded_context = tokenizer.encode(context)
    if len(encoded_context) > max_context_length:
        encoded_context = encoded_context[:max_context_length]
        context = tokenizer.decode(encoded_context, skip_special_tokens=True)

    if len(context) > max_context_length:
        context = context[:max_context_length] + "..."

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are meant to answer queries with PROTECT initiative research from the data I have provided for you. Here is the context:\n\n{context}"},
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error: Unable to complete the query due to quota limitations."

print(query("What kinds of studies have been done by PROTECT for PFAs? And name some of those papers"))
