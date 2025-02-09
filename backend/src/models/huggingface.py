from transformers import pipeline
import sys
sys.path.append("./")
from src.models.abstract import AbstractModel
from utils.device import get_device
from utils.messages import MessageHistory
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BatchEncoding
from transformers.generation import GenerateDecoderOnlyOutput
import transformers
import torch
from typing import Union
from utils.device import get_device
from enum import Enum

class ModelNames(Enum):
    llama3_2_1B = "meta-llama/Llama-3.2-1B-Instruct"


class HuggingfaceModel(AbstractModel):

    def __init__(self, model_name : str,  max_tokens : int, temperature : float):
        super().__init__(max_tokens, temperature)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = get_device()
        self.transformer.to(self.device)


    def to_prompt(self, input : Union[str, MessageHistory]) -> str:
        if type(input) is str:
            prompt = input
        else:
            prompt = input.to_prompt(self.tokenizer)
        return prompt

    def __call__(self, input : Union[str, MessageHistory]) -> str:
        prompt = self.to_prompt(input)
        gen_out = self.generate(prompt)
        return self.tokenizer.batch_decode(gen_out.sequences, skip_special_tokens=True)[0]
    
    def generate(self, prompt : str) -> GenerateDecoderOnlyOutput:
        input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output : GenerateDecoderOnlyOutput = self.transformer.generate(**input,max_new_tokens = self.max_tokens, output_logits = True, return_dict_in_generate = True, pad_token_id=self.tokenizer.eos_token_id, return_legacy_cache=True)
        output.sequences = output.sequences[:,input["input_ids"].shape[1]:]
        return output

    
    def next_probabilities(self, input : Union[str, MessageHistory], top_k : int = 5) -> dict[str, float]:
        prompt = self.to_prompt(input)
        gen_out = self.generate(prompt)
        softmax = torch.softmax(gen_out.logits[0], dim = 1)
        selected = torch.topk(softmax, dim = 1, k = top_k)        
        out_dict = {}
        pairs = zip(selected.values[0], selected.indices[0])
        for value, index in pairs:
            out_dict[self.tokenizer.batch_decode([index])[0]] = value.item()
        return out_dict





                        