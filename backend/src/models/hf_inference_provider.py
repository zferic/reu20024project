import os
from typing import Optional, List, Dict, Any
from huggingface_hub import InferenceClient
from backend.src.models.abstract import AbstractModel


class HFInferenceProviderModel(AbstractModel):
    """
    Hugging Face Inference Providers wrapper using huggingface_hub.InferenceClient
    Supports chat completions (and streaming).
    """

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
        token: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

        hf_token = token or os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN is not set. Export HF_TOKEN in environment variables.")

        # InferenceClient automatically routes to providers supported by that model
        self.client = InferenceClient(token=hf_token)

    def __call__(self, prompt: str) -> str:
        """
        Your Generator passes a single full prompt string formatted as llama-chat.
        We'll send it as one user message.
        """
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return completion.choices[0].message.content

    def stream(self, prompt: str):
        """
        Yields incremental text chunks (strings).
        Used by /generate_stream
        """
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True,
        )
        for event in stream:
            # event is a ChatCompletionStreamOutput-like object
            # It usually contains choices[0].delta.content
            try:
                delta = event.choices[0].delta
                if delta and getattr(delta, "content", None):
                    yield delta.content
            except Exception:
                # fail soft
                continue

    def next_probabilities(self, prompt: str):
        """
        Some models (like llama.cpp) can return next-token probabilities.
        HF Inference Providers chat completions do not expose token-level probabilities
        in this wrapper, so we raise a clear error.
        """
        raise NotImplementedError("next_probabilities is not supported for HFInferenceProviderModel.")

