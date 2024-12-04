import pandas as pd
import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

def read_excel_file(excel_path):
    """
    Read the Excel file, standardize column names, and drop unnamed columns.
    """
    sheet_names = pd.ExcelFile(excel_path).sheet_names
    print("Sheet names:", sheet_names)
    df_sheet = pd.read_excel(excel_path, sheet_name=sheet_names[0])
    
    df_sheet.columns = df_sheet.columns.str.strip().str.lower()
    print("Standardized Columns in the DataFrame:", df_sheet.columns.tolist())
    
    unnamed_cols = [col for col in df_sheet.columns if col.startswith('unnamed')]
    if unnamed_cols:
        df_sheet = df_sheet.drop(columns=unnamed_cols)
        print(f"Dropped unnamed columns: {unnamed_cols}")
    else:
        print("No unnamed columns to drop.")
    
    print("Final Columns after Dropping Unnamed:", df_sheet.columns.tolist())
    return df_sheet


def count_words_per_column(df):
    """
    Count the number of words in each column of the DataFrame.
    """
    def count_words(cell):
        if isinstance(cell, str):
            return len(cell.split())
        return 0
    words_by_column = df.applymap(count_words).sum()
    print("Total words by column:")
    print(words_by_column)
    return words_by_column

def combine_columns(df, columns, new_column='combined'):
    """
    Combine specified columns into a new column by concatenating their string representations.
    """
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f"The following columns are missing in the DataFrame: {missing_cols}")
    
    df[new_column] = df[columns].apply(
        lambda x: ' '.join(
            [f"{col.capitalize()}: {str(x[col])}" for col in columns]
        ),
        axis=1
    )
    return df

def initialize_model_and_tokenizer(model_name='sentence-transformers/all-distilroberta-v1'):
    """
    Initialize the tokenizer and model from transformers.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    return tokenizer, model, device

def chunk_text(text, tokenizer, max_tokens=512):
    """
    Split text into chunks of max_tokens using the tokenizer.
    """
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokenizer.decode(tokens[i:i+max_tokens])
        chunks.append(chunk)
    return chunks

def get_embedding_safe(text, tokenizer, model, device):
    """
    Get the embedding of the text using the model, safely handling exceptions.
    """
    if isinstance(text, str):
        text = text.replace("\n", " ")
        try:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)[0].cpu().numpy()
            return embedding
        except Exception as e:
            print(f"An error occurred while getting embedding: {e}")
            return None
    else:
        return None

def get_query_embedding(question, tokenizer, model, device):
    """
    Get the embedding of the query question.
    """
    print("Received question:", question)
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    question_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
    print("Question embedding shape:", question_embedding.shape)
    return question_embedding

def cosine_similarity(vec1, vec2):
    """
    Compute the cosine similarity between two vectors.
    """
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        print("Warning: Zero vector detected in embeddings.")
        return -3333
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def retrieve_closest_text(question, embeddings_df, text_df, tokenizer, model, device, section="combined", top_n=3):
    """
    Retrieve the closest text to the question based on cosine similarity.
    """
    question_embedding = get_query_embedding(question, tokenizer, model, device)
    
    def calculate_similarity(page_embedding):
        try:
            if page_embedding is None:
                return -3333
            return cosine_similarity(question_embedding, page_embedding)
        except Exception as e:
            print(f"Error evaluating embedding: {e}")
            return -222
    
    similarities = embeddings_df[section].apply(calculate_similarity)
    top_indices = similarities.nlargest(top_n).index
    top_texts = text_df.loc[top_indices, section]
    
    closest_text_combined = " | ".join([f"Concept {i + 1} - {text}" for i, text in enumerate(top_texts.tolist())])
    return closest_text_combined

def compute_embeddings(df, columns_to_process, tokenizer, model, device):
    """
    Compute embeddings for specified columns in the DataFrame.
    """
    embeddings_dict = {col: [] for col in columns_to_process}
    print("Starting embedding computation...")
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Rows"):
        for col in columns_to_process:
            text = row[col]
            embedding = get_embedding_safe(text, tokenizer, model, device)
            embeddings_dict[col].append(embedding)
    embeddings_df = pd.DataFrame(embeddings_dict)

    for col in columns_to_process:
        embeddings_df[col] = embeddings_df[col].apply(lambda emb: emb.tolist() if isinstance(emb, np.ndarray) else emb)
    return embeddings_df

def load_llama_model(model_name="unsloth/Llama-3.2-3B-Instruct-bnb-4bit", max_seq_length=10000, cache_dir=r'C:\Users\tiahi\PROTECTRAG\Research-LLM\models\\'):
    """
    Load the Llama3 model and its tokenizer for inference.
    """
    from unsloth import FastLanguageModel  
    import os
    
    os.makedirs(cache_dir, exist_ok=True)
    
    model = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        cache_dir=cache_dir,
        device_map="auto"
    )
    model = FastLanguageModel.for_inference(model)
    tokenizer = model.tokenizer  
    return model, tokenizer

def generate_text(text, model, tokenizer):
    """
    Generate text using the model based on input text.
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1028)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def return_wrapped_text(text, width=200):
    """
    Wrap the given text to the specified width.
    """
    import textwrap
    wrapped_text = "\n".join(
        [textwrap.fill(line, width=width) if len(line) > width else line for line in text.splitlines()]
    )
    return wrapped_text

def main():
    excel_path = "C:\\Users\\tiahi\\PROTECTRAG\\Research-LLM\\all_texts.xlsx"
    columns_to_combine = ['title', 'abstract', 'introduction', 'results', 'discussion', 'conclusion']
    columns_to_process = ["title", "authors", "publication_date", "introduction", "results", "discussion", "conclusion", "combined"]
    section_for_similarity = "combined"
    top_n = 5

    df_sheet = read_excel_file(excel_path)

    try:
        df_sheet = combine_columns(df_sheet, columns_to_combine, new_column='combined')
        print("'combined' column created successfully.")
    except KeyError as e:
        print(e)
        return  

    count_words_per_column(df_sheet)

    tokenizer_embed, model_embed, device_embed = initialize_model_and_tokenizer()

    embeddings_df = compute_embeddings(df_sheet, columns_to_process, tokenizer_embed, model_embed, device_embed)

    text_df = df_sheet

    question1 = "What concepts are being studied related to gestational diabetes?"

    closest_abstract = retrieve_closest_text(
        question1, embeddings_df, text_df, tokenizer_embed, model_embed, device_embed,
        section=section_for_similarity, top_n=top_n
    )

    print("Closest Abstract:", closest_abstract)

    from openai import OpenAI

    api_key = "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"  
    client = OpenAI(api_key=api_key)
    template = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
{system_msg}<|eot_id|>

<|start_header_id|>context<|end_header_id|>
{context}<|eot_id|>

<|start_header_id|>user<|end_header_id|>
{question}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""

    system_msg = (
        "You are a helpful AI assistant for answering questions based on the context provided. "
        "The context I provide includes the title, authors, publication date, results, "
        "discussion and conclusions of research studies for PROTECT. "
        "Each concept starts with Concept 1, Concept 2, Concept 3, etc. and is separated by a pipe character. "
        "Include the titles of the paper and summary of results in responses."
    )

    context = closest_abstract
    import re
    context = re.sub(r'\s+', ' ', context)
    question = question1

    formatted_string = template.format(system_msg=system_msg, context=context, question=question)

    response = client.chat.completions.create(
        model="gpt-4o-mini",  
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": formatted_string}
        ]
    )

    assistant_response = response.choices[0].message.content
    print("Assistant Response from OpenAI:")
    print(assistant_response)


if __name__ == "__main__":
    main()
