import os
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import textwrap
import time
from transformers import AutoTokenizer, AutoModel
import torch

# Initialize the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/allenai-specter', clean_up_tokenization_spaces=True)
model = AutoModel.from_pretrained('sentence-transformers/allenai-specter')

api_key = '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'
client = OpenAI(api_key=api_key)

# Send model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load data
embeddings_df = pd.read_csv(r"/media/zman/extrahd/reu20024project/scraping/embeddings_prime.csv")
#mbeddings_df.fillna("[]", inplace=True)  # Replace NaN with an empty list string
fallback_embedding = str(list(np.zeros(768)))  # 768-dimensional zero vector
embeddings_df = embeddings_df.fillna(fallback_embedding)
print("Embeddings DataFrame sample:", embeddings_df.head())  # Debug: Check embeddings format


complete_texts_df = pd.read_csv(r"/media/zman/extrahd/reu20024project/scraping/final_texts.csv")

# Columns to process
columns_to_process = ["title", "authors", "publication_date", "abstract", "introduction", "results", "discussion", "conclusion"]

# Function to split text into chunks
def chunk_text(text, max_tokens=512):
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunks.append(" ".join(words[start:end]))
        start = end
    
    return chunks

# Query function
def query(question, max_context_length=512):
    print("Received question:", question)  # Debug: question input
    question_chunks = chunk_text(question, max_tokens=512)  

    question_embeddings = []
    for chunk in question_chunks:
        print("Processing chunk:", chunk[:100])  # Debug: chunk of text being processed
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=512)

        question_embedding = model(**inputs).last_hidden_state.mean(dim=1).detach().cpu().numpy().flatten()
        question_embeddings.append(question_embedding)
        
    question_embedding = np.mean(question_embeddings, axis=0)
    print("Question embedding shape:", question_embedding.shape)  # Debug: shape of question embedding

    # Define function to compute similarity
    def fn2(question_embedding, page_embedding):
        if page_embedding is None or isinstance(page_embedding, float):  
            print("Invalid page embedding detected:", page_embedding)  # Debug
            return -np.inf
        try:
            page_embedding = np.array(eval(page_embedding), dtype=np.float32)
        except Exception as e:
            print("Error processing page embedding:", page_embedding, "Error:", e)  # Debug
            return -np.inf
        return np.dot(page_embedding, question_embedding)
    
        # Define function to compute cosine similarity
    def fn(question_embedding, page_embedding):
        if page_embedding is None or isinstance(page_embedding, float):  
            print("Invalid page embedding detected:", page_embedding)  # Debug
            return -np.inf
        try:
            # Convert the page embedding from string to a numpy array
            page_embedding = np.array(eval(page_embedding), dtype=np.float32)
            
            # Calculate cosine similarity
            dot_product = np.dot(page_embedding, question_embedding)
            magnitude_page = np.linalg.norm(page_embedding)
            magnitude_question = np.linalg.norm(question_embedding)
            
            # Handle case where either vector has zero magnitude (to avoid division by zero)
            if magnitude_page == 0 or magnitude_question == 0:
                print("Zero magnitude detected in embeddings")  # Debug
                return -np.inf
            
            cosine_similarity = dot_product / (magnitude_page * magnitude_question)
            return cosine_similarity
        except Exception as e:
            print("Error processing page embedding:", page_embedding, "Error:", e)  # Debug
            return -np.inf

    distances = []
    
    columns_to_process = ['abstract']
    for col in columns_to_process:
        print(f"Processing column: {col}")  # Debug: current column being processed
        
        # Calculate distances for the column
        col_distance_series = embeddings_df[col].apply(lambda x: fn(question_embedding, x))
        
        # Get top 5 distances and their indices
        top_5_distances = col_distance_series.nlargest(5)
        print(f"Top distances for {col}: {top_5_distances.values}")  # Debug: top 5 distances for column
        
        # Retrieve and print corresponding text for the top 5 distances
        top_5_indices = top_5_distances.index
        top_5_texts = complete_texts_df.loc[top_5_indices, col]
        
        print(f"Top texts for {col}:")
        for i, (distance, text) in enumerate(zip(top_5_distances.values, top_5_texts), 1):
            if isinstance(text, str):  # Check if text is a valid string
                print(f"{i}. Distance: {distance}, Text: {text[:200]}")  # Print first 200 chars of each text for readability
            else:
                print(f"{i}. Distance: {distance}, Text: [Invalid or missing text]")  # Handle NaN or invalid entries
        
        distances.append(top_5_distances)
    combined_distances = sum(distances)
    combined_distances.sort_values(ascending=True, inplace=True)
    print("Top combined distances:", combined_distances.head())  # Debug: combined top distances
        
    # Get top 4 results
    top_four_indices = combined_distances.index[:4]
    print("Top four indices:", top_four_indices.tolist())  # Debug: indices of top results

    contexts = []
    for col in columns_to_process:
        text_series = complete_texts_df.loc[top_four_indices, col]
        contexts.extend(text_series.dropna().astype(str).tolist())

    context = "\n\n".join(contexts)
    print("Generated context (truncated):", context[:500])  # Debug: first 500 characters of context

    if len(context) > max_context_length:
        context = context[:max_context_length] + "..."

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are meant to answer queries about the PROTECT research center from the context I have provided for you. Here is the context:\n\n{context}"},
                {"role": "user", "content": question}
            ]
        )
        answer = response.choices[0].message.content
        print("Generated answer:", answer)  # Debug: generated answer
        return answer
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error: Unable to complete the query due to quota limitations."

# Interactive loop for querying
while(True):
    question = input("****Enter a query:")
    print(query(question))
