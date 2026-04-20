from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
from typing import Union
from backend.utils.device import get_device
from enum import Enum
from .abstract import AbstractModel
from utils.messages import MessageHistory
from llama_cpp import Llama

class ModelNames(Enum):
    llama3_2_1B = "meta-llama/Llama-3.2-1B-Instruct"
    llama3_2_3B = "meta-llama/Llama-3.2-3B-Instruct"

class HuggingfaceModel(AbstractModel):
    def __init__(self, model_name: str, max_tokens: int, temperature: float, use_4bit: bool = True):
        # 1. Initialize base parameters
        super().__init__(max_tokens, temperature)
        self.device = get_device()
        self.use_4bit = use_4bit and self.device == "cpu"  
        self.response_cache = {}  
        
        # 2. Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 3. Configure quantization if needed
        quantization_config = None
        if self.use_4bit:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                )
            except Exception as e:
                print(f"Failed to create quantization config: {e}")
                quantization_config = None
        
        # 4. Set up model loading parameters
        model_kwargs = {
            "torch_dtype": torch.float32,
            "low_cpu_mem_usage": True,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        # 5. Load the model
        try:
            self.transformer = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
        except Exception as e:
            print(f"Error loading model with quantization: {e}")
            self.transformer = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
        
        # 6. Move model to device and optimize
        self.transformer.to(self.device)
        
        if self.device == "cpu":
            if hasattr(torch, "set_num_threads"):
                num_threads = max(1, os.cpu_count() // 2) if os.cpu_count() else 4
                torch.set_num_threads(num_threads)
            
            for param in self.transformer.parameters():
                param.requires_grad = False

    def to_prompt(self, input: Union[str, MessageHistory]) -> str:
        # 1. Convert input to prompt string
        if isinstance(input, str):
            return input
        return input.to_prompt(self.tokenizer)

    def __call__(self, input: Union[str, MessageHistory]) -> str:
        # 1. Convert input to prompt
        prompt = self.to_prompt(input)
        
        # 2. Check cache
        if prompt in self.response_cache:
            return self.response_cache[prompt]
        
        # 3. Generate response
        gen_out = self.generate(prompt)
        response = self.tokenizer.batch_decode(gen_out.sequences, skip_special_tokens=True)[0]
        
        # 4. Cache and return
        self.response_cache[prompt] = response
        return response
    
    def generate(self, prompt: str):
        # 1. Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 2. Generate text
        with torch.inference_mode():
            output = self.transformer.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
                use_cache=True,
                num_beams=1,
            )
        
        # 3. Extract new tokens
        output.sequences = output.sequences[:, inputs["input_ids"].shape[1]:]
        return output

    def next_probabilities(self, input: Union[str, MessageHistory], top_k: int = 5) -> dict[str, float]:
        # 1. Get prompt and generate
        prompt = self.to_prompt(input)
        gen_out = self.generate(prompt)
        
        # 2. Calculate probabilities
        softmax = torch.softmax(gen_out.scores[0], dim=1)
        selected = torch.topk(softmax, dim=1, k=top_k)
        
        # 3. Convert to dictionary
        out_dict = {}
        pairs = zip(selected.values[0], selected.indices[0])
        for value, index in pairs:
            out_dict[self.tokenizer.batch_decode([index])[0]] = value.item()
        return out_dict



# -----------------------------
# NEW: llama.cpp model backend
# -----------------------------
class LlamaCppModel(AbstractModel):
    def __init__(self, model_path, max_tokens=512, temperature=0.3, n_threads=6):
        super().__init__(max_tokens, temperature)
        from llama_cpp import Llama
        self.model = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=n_threads,
            use_mlock=True,
            verbose=False
        )

    def __call__(self, prompt: str) -> str:
        output = self.model(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["</s>"]
        )
        return output["choices"][0]["text"]

    # 🧩 Add this to satisfy the abstract interface
    def next_probabilities(self, input, top_k: int = 5) -> dict[str, float]:
        return {}
