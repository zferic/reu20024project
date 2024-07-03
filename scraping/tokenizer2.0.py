from transformers import AutoTokenizer
import os
import re
from collections import Counter
import torch
from torch.utils.data import Dataset, ConcatDataset
import matplotlib.pyplot as plt

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

def tokenize_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

            abstract_start = text.find("### Abstract ###")
            if abstract_start == -1:
                print(f"No abstract found in {file_path}")
                return []

            abstract_start += len("### Abstract ###")
            abstract_end = text.find("###", abstract_start)
            if abstract_end == -1:
                print(f"No ending for abstract found in {file_path}")
                return []

            abstract_text = text[abstract_start:abstract_end].strip()
            if not abstract_text:
                print(f"Empty abstract in {file_path}")
                return []

            tokens = tokenizer.tokenize(abstract_text)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            
            return token_ids
    except Exception as e:
        print(f"Error tokenizing file {file_path}: {e}")
        return []

def tokenize_directory(directory):
    tokenized_texts = {}
    for file_name in os.listdir(directory):
        if file_name.endswith("_sections.txt"):
            file_path = os.path.join(directory, file_name)
            tokens = tokenize_file(file_path)
            tokenized_texts[file_name] = tokens
    return tokenized_texts

class AutoregressiveDataset(Dataset):
    def __init__(self, tokens):
        self.tokens = tokens
        self.data = self.create_sequences(tokens)

    def create_sequences(self, tokens):
        sequences = []
        for i in range(len(tokens) - 1): 
            input_sequence = tokens[:i + 1]
            target_token = tokens[i + 1]
            sequences.append((input_sequence, target_token))
        return sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_sequence, target_token = self.data[idx]
        input_tensor = torch.tensor(input_sequence, dtype=torch.long)
        target_tensor = torch.tensor(target_token, dtype=torch.long)
        return input_tensor, target_tensor

if __name__ == "__main__":
    directory = r"C:\Users\tiahi\NSF REU\tokenizing\downloaded_txts"
    tokenized_texts = tokenize_directory(directory)
    
    cumulative_token_counts = Counter()
    all_datasets = []

    for file_name, tokens in tokenized_texts.items():
        if tokens:
            try:
                dataset = AutoregressiveDataset(tokens)
                all_datasets.append(dataset)

                alphanumeric_tokens = [token for token in tokens if tokenizer.convert_ids_to_tokens([token])[0].isalnum()]
                cumulative_token_counts.update(alphanumeric_tokens)
            except Exception as e:
                print(f"Error creating dataset for {file_name}: {e}")

    if all_datasets:
        combined_dataset = ConcatDataset(all_datasets)
        torch.save(combined_dataset, 'combined_dataset.pt')
    else:
        print("No valid datasets to combine.")

    top_10_tokens = cumulative_token_counts.most_common(10)
    top_10_token_ids, top_10_counts = zip(*top_10_tokens)
    top_10_token_strs = [tokenizer.convert_ids_to_tokens([token_id])[0] for token_id in top_10_token_ids]

    plt.figure(figsize=(10, 6))
    plt.bar(top_10_token_strs, top_10_counts)
    plt.xlabel('Tokens')
    plt.ylabel('Frequency')
    plt.title('Top 10 Alphanumeric Tokens Across All Articles')
    plt.show()
