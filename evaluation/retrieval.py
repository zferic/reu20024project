import openai
from openai_utils import *
from typing import Optional
import math


def print_messages(messages):
    for m in messages:
        print(m)

class RetrievalBenchmark:
    """
    Allows for the benchmarking of retrieval through checks that retrieved documents contain the answer to a given question. This class is constructed using the prompt to be answered.

    RetrievalBenchmark("What is 2 + 2").answeredBy("Two plus two is 4"]) > 0.99

    RetrievalBenchmark("What indicates that the molecular genotoxicity assay is suitable for detection")
        .answeredBy("The proposed molecular endpoints derived from the toxicogenomics assays, namely TELI 
                    (Transcriptional Effect Level Index) and PELI (Protein Effect Level Index), correlated well with the phenotypic DNA damage endpoints from comet tests,
                      suggesting that the molecular genotoxicity assay is suitable for genotoxicity detection.") > 0.99
    """

    

    MODEL = "gpt-4o-mini"


    def __init__(self, prompt : str):
        self.prompt = prompt
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


    def _get_true_probability(self, messages : str) -> float:
        """
        Private helper method that calculates the probability of 'true' given the provided messages
        """


        res = self.client.chat.completions.create(
            model = RetrievalBenchmark.MODEL,
            messages= messages,
            logprobs = True,
            temperature=0,
            top_logprobs=5
        )

        logprob_objs = res.choices[0].logprobs.content[0].top_logprobs
        token_logprobs = []
        for val in logprob_objs:
            token_logprobs.append((val.token, val.logprob))
        return self._logprobs_to_true_prob(token_logprobs)



    def answeredBy(self, document : str) -> float:
        """
        Returns a probability, from 0 to 1, that the given document is needed to answer the user's prompt.
        """

        SYS_PROMPT = """I want you to help benchmark my Retrieval Augmented Generation pipeline. I'm going to send you a user's prompt, or question, with a retrieved document that should contain information needed
to provide a user a response. If the given document is in fact needed, reply with the word 'true'. If it is not, reply with 'false'. You should consider whether the provided prompt can be answered by common knowledge or not in your evaluation,
only consider whether the document has the relevant information or not. Pay attention if the given document is relevant to the prompt but does not actually provide an answer to the user's question, this should be marked as 'false'.""" 

        def make_user_message(given_doc : str, given_question : str):
            return f"Here is the document:\n{given_doc}\n\nHere is the question:\n{given_question}\n\nAnswer 'true' if the document contains information needed to answer the users question, and 'false' if it does not."


        EXAMPLES = [
            # [document, question, answer]
            ["Exclusion criteria included: use of tobacco products other than cigarettes (e.g., cigars, e-cigarettes, chewing tobacco), the presence of chronic liver or kidney disease, as these conditions could affect toxin metabolism, and pregnancy or breastfeeding", 
             "What exclusion criteria were considered for the study?",
             "true"],
             
             ["Participants were classified as current smokers, former smokers, or never-smokers based on self-report. Current smoking status was confirmed by measurement of plasma cotinine levels (>10 ng/mL). Pack-years were computed as (packs smoked per day) × (years of smoking).",
              "Does the metabolite cotinine have any psychoactive effect in smokers?",
              "false"]
        ]


        messages = []
        add_sys_msg(messages, SYS_PROMPT)
        for ex in EXAMPLES:
            add_user_msg(messages, make_user_message(ex[0], ex[1]))
            add_model_msg(messages, ex[2])
        add_user_msg(messages, make_user_message(document, self.prompt))
        print(messages)
        return self._get_true_probability(messages)



        


