import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# Set cache directory paths to scratch
os.environ['HF_HOME'] = '/media/zman/extrahd'  # Hugging Face model cache
os.environ['HF_DATASETS_CACHE'] = '/media/zman/extrahd'  # Datasets cache

max_seq_length = 2048
url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
path = "/media/zman/extrahd/reu20024project/qastuff/output_file.jsonl"

dataset = load_dataset("json", data_files = {"train" : path}, split = "train")

# 2. Load Llama3 model (replace unsloth with HF llama)
model_name = "meta-llama/Llama-3.1-8B" # Adjust model to a regular Llama HF model



# Use token if needed (replace with your token)
token = "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"

# Load the tokenizer and model, using bitsandbytes for 4bit quantization
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,  # Using bitsandbytes 4-bit quantization
    torch_dtype=torch.float16,
    device_map="auto"
)

# 3 Before training
def generate_text(text):
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
    outputs = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print("Before training\n")
generate_text("<human>: List the top 5 most popular movies of all time.\n<bot>: ")

# 4. Apply LoRA for efficient fine-tuning
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none"
)

model = get_peft_model(model, lora_config)

# 5. Training setup
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=60,
    fp16=True,  # Use FP16 to save memory
    logging_steps=1,
    output_dir="./outputs",
    optim="adamw_bnb_8bit",  # Use bitsandbytes AdamW optimizer
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# 6. Train the model
trainer.train()

# 7. After training
print("\n ######## \nAfter training\n")
generate_text("<human>: List the top 5 most popular movies of all time.\n<bot>: ")
