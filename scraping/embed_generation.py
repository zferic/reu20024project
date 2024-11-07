import os
import pandas as pd
import numpy as np
import textwrap
import time
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm  

print("Initializing tokenizer and model...")
start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1', clean_up_tokenization_spaces=True)
model = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1')

# Send model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

init_time = time.time() - start_time
print(f"Initialization completed in {init_time:.2f} seconds.\n")



def chunk_text(text, max_tokens=512):
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokenizer.decode(tokens[i:i+max_tokens])
        chunks.append(chunk)
    return chunks

def get_embedding_safe(text):
    if isinstance(text, str):
        text = text.replace("\n", " ")
        try:
            #text_chunks = chunk_text(text, max_tokens=512)
            embeddings = []
            #print(text_chunks)
            #for chunk in text_chunks:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
            #embedding = outputs.pooler_output[0].cpu().numpy() 
            embedding = outputs.last_hidden_state.mean(dim=1)[0].cpu().numpy() 
                #embedding = outputs.pooler_output[0].numpy()  
            embeddings.append(embedding)
            return embeddings[0]
            #return np.mean(embeddings, axis=0)
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    else:
        return None

data_loading_start = time.time()
print("Loading DataFrame from CSV...")
df = pd.read_csv(r"/media/zman/extrahd/reu20024project/scraping/final_texts.csv")
data_loading_time = time.time() - data_loading_start
print(f"DataFrame loaded in {data_loading_time:.2f} seconds.\n")

columns_to_process = ["title", "authors", "publication_date", "abstract", "introduction", "results", "discussion", "conclusion"]

embeddings_dict = {col: [] for col in columns_to_process}

print("Starting embedding computation...")
processing_start = time.time()
for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Rows"):
    for col in columns_to_process:
        text = row[col]
        embedding = get_embedding_safe(text)
        embeddings_dict[col].append(embedding)
processing_time = time.time() - processing_start
print(f"Embedding computation completed in {processing_time:.2f} seconds.\n")

print("Creating embeddings DataFrame...")
embeddings_df = pd.DataFrame(embeddings_dict)

print("Converting embeddings to lists...")
conversion_start = time.time()
for col in columns_to_process:
    embeddings_df[col] = embeddings_df[col].apply(lambda emb: emb.tolist() if isinstance(emb, np.ndarray) else emb)
conversion_time = time.time() - conversion_start
print(f"Conversion completed in {conversion_time:.2f} seconds.\n")

print("Saving embeddings to 'embeddings_prime.csv'...")
save_start = time.time()

embeddings_df.to_csv("embeddings_prime.csv", index=False)
save_time = time.time() - save_start
print(f"Saved embeddings to 'embeddings_prime.csv' in {save_time:.2f} seconds.\n")

total_time = init_time + data_loading_time + processing_time + conversion_time + save_time
print(f"Total elapsed time: {total_time:.2f} seconds.")
