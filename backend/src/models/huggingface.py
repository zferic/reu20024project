from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Union
from backend.utils.device import get_device
from enum import Enum
from .abstract import AbstractModel
from utils.messages import MessageHistory

class ModelNames(Enum):
    llama3_2_1B = "meta-llama/Llama-3.2-1B-Instruct"

class HuggingfaceModel(AbstractModel):
    def __init__(self, model_name: str, max_tokens: int, temperature: float):
        super().__init__(max_tokens, temperature)
        self.device = get_device()
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )
        self.transformer.to(self.device)
        
        if self.device == "cuda":
            self.transformer = torch.compile(self.transformer)  # Optional: Enable torch.compile for better performance

    def to_prompt(self, input: Union[str, MessageHistory]) -> str:
        if isinstance(input, str):
            return input
        return input.to_prompt(self.tokenizer)

    def __call__(self, input: Union[str, MessageHistory]) -> str:
        prompt = self.to_prompt(input)
        gen_out = self.generate(prompt)
        return self.tokenizer.batch_decode(gen_out.sequences, skip_special_tokens=True)[0]
    
    def generate(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self.transformer.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True
            )
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





                        