from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import time
import os
from typing import Union, Dict, Any
from backend.utils.device import get_device
from enum import Enum
from .abstract import AbstractModel
from utils.messages import MessageHistory

class ModelNames(Enum):
    llama3_2_1B = "meta-llama/Llama-3.2-1B-Instruct"

class HuggingfaceModel(AbstractModel):
    def __init__(self, model_name: str, max_tokens: int, temperature: float, use_4bit: bool = True):
        super().__init__(max_tokens, temperature)
        self.device = get_device()
        self.use_4bit = use_4bit and self.device == "cpu"  # Only use 4-bit quantization on CPU
        self.response_cache: Dict[str, str] = {}  # Cache for generated responses
        
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        start_time = time.time()
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Tokenizer loaded in {time.time() - start_time:.2f} seconds")
        
        # Configure model loading options
        start_time = time.time()
        print(f"Loading model: {model_name}")
        
        # Set up quantization config for CPU
        quantization_config = None
        if self.use_4bit:
            print("Using 4-bit quantization for CPU efficiency")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float32,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        
        # Load model with optimizations
        model_kwargs = {
            "torch_dtype": torch.float32,  # Use float32 for CPU
            "low_cpu_mem_usage": True,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        self.transformer = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Move model to device
        self.transformer.to(self.device)
        
        # Set up CPU-specific optimizations
        if self.device == "cpu":
            # Enable model CPU threading if available
            if hasattr(torch, "set_num_threads"):
                # Use half of available CPU cores for better parallelism without overloading
                num_threads = max(1, os.cpu_count() // 2) if os.cpu_count() else 4
                torch.set_num_threads(num_threads)
                print(f"Set PyTorch to use {num_threads} CPU threads")
            
            # Disable gradient computation for inference
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")

    def to_prompt(self, input: Union[str, MessageHistory]) -> str:
        if isinstance(input, str):
            return input
        return input.to_prompt(self.tokenizer)

    def __call__(self, input: Union[str, MessageHistory]) -> str:
        prompt = self.to_prompt(input)
        
        # Check cache first
        if prompt in self.response_cache:
            print("Using cached response")
            return self.response_cache[prompt]
        
        # Generate response
        start_time = time.time()
        gen_out = self.generate(prompt)
        response = self.tokenizer.batch_decode(gen_out.sequences, skip_special_tokens=True)[0]
        print(f"Generation completed in {time.time() - start_time:.2f} seconds")
        
        # Cache response
        self.response_cache[prompt] = response
        
        return response
    
    def generate(self, prompt: str):
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate with optimized settings for CPU
        with torch.inference_mode():
            output = self.transformer.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
                # CPU optimizations
                use_cache=True,  # Enable KV caching
                num_beams=1,     # Disable beam search for faster generation
            )
        
        # Extract only the new tokens (exclude input tokens)
        output.sequences = output.sequences[:, inputs["input_ids"].shape[1]:]
        return output

    def next_probabilities(self, input: Union[str, MessageHistory], top_k: int = 5) -> dict[str, float]:
        prompt = self.to_prompt(input)
        gen_out = self.generate(prompt)
        softmax = torch.softmax(gen_out.scores[0], dim=1)
        selected = torch.topk(softmax, dim=1, k=top_k)
        out_dict = {}
        pairs = zip(selected.values[0], selected.indices[0])
        for value, index in pairs:
            out_dict[self.tokenizer.batch_decode([index])[0]] = value.item()
        return out_dict





                        