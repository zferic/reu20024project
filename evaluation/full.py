import openai
import sys
sys.path.append("./")
from evaluation.judge import AbstractJudge
from utils.messages import *



class GenerationJudge(AbstractJudge):
    """
    Allows for the benchmarking if generated answers to questions are correct based on a predefined context/cannonical answer.

    GenerationJudge(question = "What are TELI and PELI?"
                    context = "The proposed molecular endpoints derived from the toxicogenomics assays, namely TELI (Transcriptional Effect Level Index) and PELI (Protein Effect Level Index), correlated well with the phenotypic DNA damage endpoints from comet tests, suggesting that the molecular genotoxicity assay is suitable for genotoxicity detection."
                    ).qualityOf("TELI (Transcriptional Effect Level Index) and PELI (Protein Effect Level Index) are proposed molecular endpoints derived from the toxicogenomics assays.")
    """

    

    MODEL = "gpt-4o-mini"

    def __init__(self, question : str, context : str):
        super().__init__(GenerationJudge.MODEL)
        self.question = question
        self.context = context
        self.client = openai.OpenAI()

    def accuracyOf(self, answer : str) -> float:
        """
        Returns a probability, from 0 to 1, that the given answer is accurate, relevant, and high quality, based on the context. 
        """

        SYS_PROMPT = ("I want you to help benchmark the performance of an LLM. I'm going to provide you with a question, and a piece of context that contains the answer.\n"
                      "I will then provide you with an LLM generated answer. If you feel the generated answer is high quality, meaning it is relevant to the question, well structured\n" 
                      "and provides a full and correct answer to the question, reply with the word 'true'. If you do not feel like the generated answer is high quality based on these critera,\n"
                      "respond with the word 'false'")

        def make_user_message(given_context : str, given_question : str, given_answer : str):
            return f"Here is the question:\n{given_question}\n\nHere is the context:\n{given_context}\n\nHere is the LLM generated answer:{given_answer}\n\nRespond with 'true' or 'false' based on the given criteria."



        EXAMPLES = [
            # [document, question, answer]
            ["Exclusion criteria included: use of tobacco products other than cigarettes (e.g., cigars, e-cigarettes, chewing tobacco), the presence of chronic liver or kidney disease, as these conditions could affect toxin metabolism, and pregnancy or breastfeeding", 
             "What exclusion criteria were considered for the study?",
             "Use of tobacco products other than cigarettes or a history of heart disease",
             "false"],
             
             ["Participants were classified as current smokers, former smokers, or never-smokers based on self-report. Current smoking status was confirmed by measurement of plasma cotinine levels (>10 ng/mL). Pack-years were computed as (packs smoked per day) × (years of smoking).",
              "What biomarker was considered in order to confirm smoking status?",
              "Within the study, smoking status was confirmed by a plasma cotinine levels over >10 ng/mL",
              "true"]
        ]


        messages = []
        add_sys_msg(messages, SYS_PROMPT)
        for ex in EXAMPLES:
            add_user_msg(messages, make_user_message(ex[0], ex[1]))
            add_model_msg(messages, ex[2])
        add_user_msg(messages, make_user_message(self.context, self.question, answer))
        return self.get_true_probability(messages)


        


