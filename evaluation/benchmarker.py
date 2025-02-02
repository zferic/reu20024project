from typing import Callable
import json
from retrieval import PromptRetrievalJudge
import sys
sys.path.append("./")
from models.abstract import AbstractModel
from langchain_core.documents import Document
from tqdm import tqdm
import time
import random
QA_PATH = "evaluation/training_qa.json"
QUESTION = "question"
CONTEXT = "context"
class Benchmarker:
    """
    Should be constructed with a function to retrieve lists of documents based on a given prompt. When 'eval' is called, 
    will run the retriever function on a list of preselected prompts, all will provide a report on the retriever functions performance.
    """

    def __init__(self, 
                 named_retrieval_functions : dict[str, Callable[[str, int], list[Document]]], 
                 judge_model : AbstractModel,
                 use_questions : int = 100):
        self.named_retrieval_functions = named_retrieval_functions
        self.judge_model = judge_model
        data = Benchmarker._load_json()
        self.test_questions = []
        self.control_dict = {}
        for j in data[:use_questions]:
            self.test_questions.append(j[QUESTION])
            self.control_dict[j[QUESTION]] = j[CONTEXT]
        self.named_retrieval_functions["Control"] = self.control_retriever
        #self.named_retrieval_functions["Random"] = self.random_retriever
        
    def eval(self) -> float:
        """
        Produces a single floating point value between 0 and 1 representing a score for the retriever. 
        This will later be expanded into something more comprehensive
        """
        SCORES = "scores"
        RUN_TIME = "run time"
        WITH_DOCS = 5
        eval_dict = {func_name : {SCORES : [], RUN_TIME: 0} for func_name in self.named_retrieval_functions}
        for question in tqdm(self.test_questions, desc = "Running Evaluation"):
            prompt = PromptRetrievalJudge(question, self.judge_model)
            for func_name in self.named_retrieval_functions:
                t1 = time.time()
                retrieved = self.named_retrieval_functions[func_name](question, WITH_DOCS)
                t = time.time() - t1
                eval_dict[func_name][SCORES].append(prompt.answeredBy(retrieved))
                eval_dict[func_name][RUN_TIME] += t
        return eval_dict

    @staticmethod
    def _load_json() -> dict[str, str]:
        import os
        print(os.listdir("."))
        with open(QA_PATH, "r") as f:
            data = json.load(f)
        return data

    def control_retriever(self, question : str, n : int) -> list[Document]:
        """
        A mock retriever that uses the source documents used to generate the questions for the evaluation. 
        This should be used as a control of what the PromptRetrievalJudge should deem as a perfect retriever
        """
        return [Document(self.control_dict[question]) for _ in range(n)]


    def identity_retriever(self, question : str, n : int) -> list[Document]:
        """
        A mock retriver that fetches the given question as the document
        """
        return [Document(question) for _ in range(n)]
    

