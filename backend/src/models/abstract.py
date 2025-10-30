from abc import ABC, abstractmethod
from typing import Union
from utils.messages import MessageHistory

class AbstractModel(ABC):

    def __init__(self, max_tokens : int, temperature : float):
        self.max_tokens = max_tokens
        self.temperature = temperature

    @abstractmethod
    def __call__(self, input : Union[str, MessageHistory]) -> str:
        """
        Returns a completion based on the given prompt.
        """
        ...


    @abstractmethod
    def next_probabilities(self, input : Union[str, MessageHistory], top_k : int = 5) -> dict[str, float]:
        """
        Returns the probabilties of the top_k most likely next tokens as a dictionary from token to probability
        """

    