from transformers import AutoTokenizer
import os
import re
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import time


tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

def tokenize_file(file_path, section_names):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

            token_ids_per_section = []

            for section_name in section_names:
                section_start = text.find(f"### {section_name} ###")
                if section_start == -1:
                    print(f"No {section_name} section found in {file_path}")
                    continue

                section_start += len(f"### {section_name} ###")
                section_end = text.find("###", section_start)
                if section_end == -1:
                    section_end = len(text)
                section_text = text[section_start:section_end].strip()

                if not section_text:
                    print(f"Empty {section_name} section in {file_path}")
                    continue

                
                tokens = tokenizer.tokenize(section_text)
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                token_ids_per_section.extend(token_ids)
            
            return token_ids_per_section
    except Exception as e:
        print(f"Error tokenizing file {file_path}: {e}")
        return []

def tokenize_directory(directory, section_names):
    tokenized_texts = {}
    for file_name in os.listdir(directory):
        if file_name.endswith("_sections.txt"):
            file_path = os.path.join(directory, file_name)
            tokens = tokenize_file(file_path, section_names)
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

def create_combined_dataset(tokenized_texts):
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
        torch.save(combined_dataset, 'complete_ds.pt')
    else:
        print("No valid datasets to combine.")
    
    return cumulative_token_counts

def measure_dataloader_speed(combined_dataset):

    dataloader = DataLoader(combined_dataset, batch_size=32, num_workers=4, shuffle=True)  

    start_time = time.time()
    for batch in dataloader:
        pass  
    end_time = time.time()

    print(f"DataLoader processing time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    directory = r"C:\Users\tiahi\NSF REU\tokenizing\downloaded_txts"
    section_names = ["Abstract", "Introduction",  "Methods", "Results", "Discussion", "Conclusion"] 
    tokenized_texts = tokenize_directory(directory, section_names)

    cumulative_token_counts = create_combined_dataset(tokenized_texts)

    measure_dataloader_speed(tokenized_texts)

    top_10_tokens = cumulative_token_counts.most_common(10)
    top_10_token_ids, top_10_counts = zip(*top_10_tokens)
    top_10_token_strs = [tokenizer.convert_ids_to_tokens([token_id])[0] for token_id in top_10_token_ids]

    plt.figure(figsize=(10, 6))
    plt.bar(top_10_token_strs, top_10_counts)
    plt.xlabel('Tokens')
    plt.ylabel('Frequency')
    plt.title('Top 10 Alphanumeric Tokens Across All Articles')
    plt.show()
