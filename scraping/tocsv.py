from transformers import AutoTokenizer
import os
import re
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

def extract_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

            text = re.sub(r"###.*?###", "", text)
            text = re.sub(r"Title:\s*|Authors:\s*|Publication Date:\s*", "", text)

            return text.strip()
    except Exception as e:
        print(f"Error extracting text from file {file_path}: {e}")
        return None

def extract_texts_from_directory(directory):
    texts = []
    for file_name in os.listdir(directory):
        if file_name.endswith("_sections.txt"):
            file_path = os.path.join(directory, file_name)
            text = extract_text(file_path)
            if text:
                texts.append(text)
    return texts

if __name__ == "__main__":
    directory = r"C:\Users\tiahi\NSF REU\tokenizing\updated_txts"
    texts = extract_texts_from_directory(directory)
    
    df = pd.DataFrame(texts, columns=["text"])
    df.to_csv("all_texts.csv", index=False)
    print(f"Saved {len(texts)} texts to 'all_texts.csv'")
