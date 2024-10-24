import os
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

max_seq_length = 2048
url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
path = "/media/zman/extrahd/reu20024project/qastuff/output_file.jsonl"

dataset = load_dataset("json", data_files = {"train" : path}, split = "train")

# 2. Load Llama3 model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

# 3 Before training
def generate_text(text, model):
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
    outputs = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print("Before training\n")
#generate_text("<human>: What is PROTECT?.\n<bot>: ")

# 4. Do model patching and add fast LoRA weights and training
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = True,
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False,  # Rank stabilized LoRA
    loftq_config = None, # LoftQ
)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    tokenizer = tokenizer,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 60,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
    ),
)
trainer.train()


# 2. Load Llama3 model
original_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)
# Call for_inference to initialize the model for inference
original_model = FastLanguageModel.for_inference(original_model)
model = FastLanguageModel.for_inference(model)


# 5. After training
# 5. After training
print("\n ######## \Before training\n")
generate_text("<human>: What is the PROTECT center?\n<bot>: ", original_model)
print("\n ######## \nAfter training\n")
generate_text("<human>: What is the PROTECT center?\n<bot>: ", model)


# 5. After training
# 5. After training
print("\n ######## \Before training\n")
generate_text("<human>: How many participants in PROTECT?\n<bot>: ", original_model)
print("\n ######## \nAfter training\n")
generate_text("<human>: How many participants in PROTECT?\n<bot>: ", model)

# 5. After training
print("\n ######## \Before training\n")
generate_text("<human>: What phalates does PROTECT study?\n<bot>: ", original_model)
print("\n ######## \nAfter training\n")
generate_text("<human>: What phalates does PROTECT study?\n<bot>: ", model)

