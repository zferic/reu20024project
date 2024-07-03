from transformers import AutoTokenizer
import os

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

def tokenize_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        
        abstract_start = text.find("### Abstract ###")
        if abstract_start == -1:
            return []

        abstract_start += len("### Abstract ###")
        abstract_end = text.find("###", abstract_start)
        abstract_text = text[abstract_start:abstract_end].strip()
        
        tokens = tokenizer.tokenize(abstract_text)
        return tokens

def tokenize_directory(directory):
    tokenized_texts = {}
    for file_name in os.listdir(directory):
         if file_name.endswith("_sections.txt"):
             file_path = os.path.join(directory, file_name)
             tokens = tokenize_file(file_path)
             tokenized_texts[file_name] = tokens
    return tokenized_texts

if __name__ == "__main__":
    directory = r"C:\Users\tiahi\NSF REU\tokenizing\downloaded_txts"
    tokenized_texts = tokenize_directory(directory)
    for file_name, tokens in tokenized_texts.items():
        print(f"Tokens for {file_name}:")
        print(tokens)
        print("---")
