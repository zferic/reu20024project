from typing import Union, Literal
from transformers import PreTrainedTokenizer, BatchEncoding


SYSTEM_ROLL = "system"
USER_ROLL = "user"
MODEL_ROLL = "assistant"


class MessageHistory:
    """
    A structure used for storing message histories
    """
    
    def __init__(self, system_prompt : str):
        self.messages = []
        self._add_msg(SYSTEM_ROLL, system_prompt)

    def to_prompt(self, tokenizer : PreTrainedTokenizer) -> str:
        return tokenizer.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
    
    def add_user_message(self, text : str):
        self._add_msg(USER_ROLL, text)

    def add_model_message(self, text : str):
        self._add_msg(MODEL_ROLL, text)

    def _add_msg(self, role : str, content : str):
        self.messages.append({"role": role, "content": content})

    def copy(self) -> "MessageHistory":
        c = MessageHistory("")
        c.messages = self.messages.copy()
        return c

def print_messages(messages):
    for m in messages:
        print(m)