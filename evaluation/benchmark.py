import openai
from typing import Optional
import math

SYSTEM_ROLL = "developer"
USER_ROLL = "user"
MODEL_ROLL = "assistant"

def print_messages(messages):
    for m in messages:
        print(m)

class CorpusBenchmarker:

    """
    A CorpusBenchmarker is capable of evaluating a variety of metrics on a given claim, (ie. an LLM's output) against a predefined corpus of source document strings.


    For example:

    Benchmarker(["James, born in 1969, grew up in Dallas, Texas"]).implies("James was born in 1969") == 1
    Benchmarker(["James, born in 1969, grew up in Dallas, Texas"]).implies("James was born in 1980") == 0
    Benchmarker(["James, born in 1969, grew up in Dallas, Texas"]).implies("James was a plumber") == 0 

    Benchmarker(["James, born in 1969, grew up in Dallas, Texas"]).contradicts("James was born in 1969") == 0
    Benchmarker(["James, born in 1969, grew up in Dallas, Texas"]).contradicts("James was born in 1980") == 1
    Benchmarker(["James, born in 1969, grew up in Dallas, Texas"]).contradicts("James was a plumber") == 0 
    
    Notice the last example for contradicts and implies has the same result, as a corpus does not need to contradict a claim to not directly imply it.

    """

    MODEL = "gpt-4o-mini"


    def __init__(self, corpus : list[str]):
        self.corpus = corpus
        self.client = openai.OpenAI()

    def _logprobs_to_true_prob(self, token_logprobs : list[tuple[str, float]]):
        """
        Given a list of token and logprob tuples, returns the probability of the word 'true' and variations
        """
        probability = 0
        for token, logprob in token_logprobs:
            if "true" in token.lower():
                probability += math.e ** logprob
        return probability



    def _document_supports(self, document : str, claim : str) -> float:
        PROMPT = """I'm going to give you a source docoument in the form of a short excerpt of text, and a seperate string of text, which I will call the statement.
I want you to determine if the statement is directly supported by the source document. You should answer with a single word, 'true' if the statement is directly supported by the source document, and 'false' if the statement
is not. You should consider the given source document as the single point of truth, and should not consider whether you think the claim on its own is correct or not on its own. Only consider a statement as supported if it is
DIRECTLY based on the given source document."""

        EXAMPLES= {

            "document": """Adam Selene was the 1st president of the United Republic of the Moon, serving from 2091 until his assassination in 2098. 
He led the Moon through it's war of independence, which took place between 2090 to 2095, serving as the Commander and Chief of the United Lunar Defence Force. Selene is remembered as a lunar hero and is revered as one its greatest leaders""",

            "claims": ["Adam Selene was the first president of the Moon", "The Moon is Earth's only natural satellite.", "The United Republic of the Moon had its war of independence from 2090 to 2095", "Adam Selene was a member of the Lunar Fedaralists."],

            "eval": ["true", "false", "true", "false"]

        }

        def make_message(document : str, claim : str):
            return f"Document:\n{document}\n\nClaim:\n{claim}\n\n"
        
        
        messages = [
            {"role": SYSTEM_ROLL, "content": PROMPT},
            {"role": USER_ROLL, "content": make_message(EXAMPLES["document"], EXAMPLES["claims"][0])}, 
            {"role": MODEL_ROLL, "content": EXAMPLES["eval"][0]}, 
            {"role": USER_ROLL, "content": make_message(EXAMPLES["document"], EXAMPLES["claims"][1])}, 
            {"role": MODEL_ROLL, "content": EXAMPLES["eval"][1]},
            {"role": USER_ROLL, "content": make_message(document, claim)},
        ]

        res = self.client.chat.completions.create(
            model = CorpusBenchmarker.MODEL,
            messages= messages,
            logprobs = True,
            top_logprobs=5
        )

        logprob_objs = res.choices[0].logprobs.content[0].top_logprobs
        token_logprobs = []
        for val in logprob_objs:
            token_logprobs.append((val.token, val.logprob))
        return self._logprobs_to_true_prob(token_logprobs)



    def supports(self, claim : str) -> float:
        """
        Returns a probability, from 0 to 1, that the given claim is directly supported by the CorpusBenchmarker's corpus
        """
        probability = 0
        for document in self.corpus:
            probability = max(probability, self._document_supports(document, claim))
        return probability

    def contradicts(self, claim : str) -> float:
        """
        Returns a probability, from 0 to 1, that the given claim is directly contradicted by the CorpusBenchmarker's corpus
        """
        ...


        


