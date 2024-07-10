import os
import re
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch

# Ensure the cache directory exists and is correctly set
cache_dir = "/media/zman/extrahd/reu20024project/.cache/huggingface/datasets"

# Define the directory containing the text files
text_files_dir = "/media/zman/extrahd/reu20024project/preprocessing/docs"

# Function to extract Abstract and Introduction from a paper
def extract_sections(text):
    abstract_match = re.search(r'### Abstract ###(.*?)### Introduction ###', text, re.DOTALL)
    introduction_match = re.search(r'### Introduction ###(.*?)(###|$)', text, re.DOTALL)
    
    abstract = abstract_match.group(1).strip() if abstract_match else ""
    introduction = introduction_match.group(1).strip() if introduction_match else ""
    
    return abstract, introduction

# Initialize lists to store the extracted sections
abstracts = []
introductions = []

# Read all text files and extract sections
for filename in os.listdir(text_files_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(text_files_dir, filename), 'r', encoding='utf-8') as file:
            text = file.read()
            abstract, introduction = extract_sections(text)
            if len(abstract.strip()) > 100:
                abstracts.append(abstract)
                introductions.append(introduction)

# Create a dataset from the extracted sections
dataset_dict = {
    "content": abstracts + introductions,
    "section": ["abstract"] * len(abstracts) + ["introduction"] * len(introductions)
}

raw_datasets = DatasetDict({
    "train": Dataset.from_dict(dataset_dict),
    "valid": Dataset.from_dict(dataset_dict)  # You can split or adjust this as needed
})

# Print the length of each dataset within the DatasetDict
print("Raw datasets length:")
print(f"Train dataset length: {len(raw_datasets['train'])}")
print(f"Valid dataset length: {len(raw_datasets['valid'])}")

# Function to print a few examples from the dataset
def print_dataset_examples(dataset, num_examples=5):
    print(f"\nShowing {num_examples} examples from the dataset:")
    for i in range(num_examples):
        print(f"Example {i + 1}:")
        print(f"Content: {dataset[i]['content'][:500]}")  # Print first 500 characters
        print(f"Section: {dataset[i]['section']}")
        print("-" * 80)

# Print examples from the training set
print_dataset_examples(raw_datasets["train"], num_examples=5)

# Tokenization as per your original script
context_length = 128
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Check if eos_token exists; if not, add it
if tokenizer.eos_token is None:
    tokenizer.add_special_tokens({'eos_token': '[EOS]'})

# Set the padding token to be the eos token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

print("pad token", tokenizer.pad_token)
print("eos token", tokenizer.eos_token)

def tokenize(element):
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
        padding='max_length'  # Ensure sequences are padded
    )
    input_batch = []
    attention_masks = []
    for length, input_ids, attention_mask in zip(outputs["length"], outputs["input_ids"], outputs["attention_mask"]):
        if length == context_length:
            input_batch.append(input_ids)
            attention_masks.append(attention_mask)
    return {"input_ids": input_batch, "attention_mask": attention_masks}

# Tokenize the datasets
tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
)

# Save the tokenized dataset to disk
tokenized_datasets.save_to_disk("/media/zman/extrahd/reu20024project/tokenized-dataset")

# Load the configuration for GPT-2
config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# Initialize the model
model = GPT2LMHeadModel(config)
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

# Initialize the data collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Test the data collator on a few examples
out = data_collator([tokenized_datasets["train"][i] for i in range(5)])

for key in out:
    print(f"{key} shape: {out[key].shape}")

# Print the original inputs and the collated outputs
for i in range(5):
    original_input_ids = tokenized_datasets["train"][i]["input_ids"]
    collated_input_ids = out["input_ids"][i].tolist()
    collated_labels = out["labels"][i].tolist()
    
    print(f"Example {i}:")
    print(f"Original Input IDs: {original_input_ids}")
    print(f"Decoded Original Input: {tokenizer.decode(original_input_ids)}")
    print(f"Collated Input IDs: {collated_input_ids}")
    print(f"Decoded Collated Input: {tokenizer.decode(collated_input_ids)}")
    print(f"Collated Labels (Targets): {collated_labels}")
    print(f"Decoded Collated Labels: {tokenizer.decode([token for token in collated_labels if token != -100])}\n")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0 only

# Set training arguments
args = TrainingArguments(
    output_dir="codeparrot-ds",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="steps",
    eval_steps=5_0,
    logging_steps=5_0,
    gradient_accumulation_steps=8,
    num_train_epochs=100,
    weight_decay=0.1,
    warmup_steps=1_00,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=5_000,
    fp16=True,
    push_to_hub=False,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
)

# Train the model
trainer.train()

# Save the model and tokenizer to a new path
#new_model_path = "codeparrot-ds-new/model"
#new_tokenizer_path = "codeparrot-ds-new/tokenizer"

#model.save_pretrained(new_model_path)
#tokenizer.save_pretrained(new_tokenizer_path)

# Function to load the model and tokenizer
def load_model(model_path, tokenizer_path):
    model = GPT2LMHeadModel(config)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


    return model, tokenizer

# Function to ask questions using the trained model
def ask_question(question, model, tokenizer):
    inputs = tokenizer.encode_plus(question, return_tensors="pt", padding=True)
    if torch.cuda.is_available():  # Ensure this if using GPU
        inputs = {key: val.to('cuda:0') for key, val in inputs.items()}
        model.to('cuda:0')
    outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=64, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Load the trained model
#model, tokenizer = load_model(new_model_path, new_tokenizer_path)

print("Trained Model Response")
question = "Wich chemicals are humans exposed to?"
response = ask_question(question, model, tokenizer)
print(response)

# Load the original model (e.g., GPT-2)
original_model_path = "gpt2"
original_tokenizer_path = "gpt2"

original_model, original_tokenizer = load_model(original_model_path, original_tokenizer_path)

# Check if eos_token exists; if not, add it

original_tokenizer.add_special_tokens({'eos_token': '[EOS]'})

# Set the padding token to be the eos token
original_tokenizer.pad_token = original_tokenizer.eos_token
original_tokenizer.pad_token_id = original_tokenizer.eos_token_id

print("pad token", original_tokenizer.pad_token)
print("eos token", original_tokenizer.eos_token)

print("Original Model Response")
response = ask_question(question, original_model, original_tokenizer)
print(response)