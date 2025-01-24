from typing import Callable
import json
from retrieval import RetrievalBenchmark
from tqdm import tqdm
QA_PATH = "training_qa.json"
QUESTION = "question"
CONTEXT = "context"
class BenchmarkRunner:
    """
    Should be constructed with a function to retrieve lists of documents based on a given prompt. When 'eval' is called, 
    will run the retriever function on a list of preselected prompts, all will provide a report on the retriever functions performance.
    """

    def __init__(self, retrieval_function : Callable[[str], list[str]], use_questions : int = 100):
        self.retrieval_function = retrieval_function
        data = BenchmarkRunner._load_json()
        self.test_questions = []
        for j in data[:use_questions]:
            self.test_questions.append(j[QUESTION])
        
    def eval(self) -> float:
        """
        Produces a single floating point value between 0 and 1 representing a score for the retriever. 
        This will later be expanded into something more comprehensive
        """
        score_list = []
        for question in tqdm(self.test_questions, desc = "Running Evaluation"):
            benchmarker = RetrievalBenchmark(question)
            retrieved = self.retrieval_function(question)
            score_list.append(benchmarker.answeredBy(retrieved))
        return sum(score_list) / len(score_list)

    @staticmethod
    def _load_json() -> dict[str, str]:
        with open(QA_PATH, "r") as f:
            data = json.load(f)
        return data

    @staticmethod
    def verify_eval(use_questions : int = 10) -> float:
        """
        Runs the benchmarker using the true pieces of context used to generate the test questions for retrievers.
        This should return a value close to 1 if RetrievalBenchmark is accurate. 
        """
        data = BenchmarkRunner._load_json()
        questions = []
        documents = []
        for j in data[:use_questions]:
            questions.append(j[QUESTION])
            documents.append([j[CONTEXT]])
        score_list = []
        for i in tqdm(range(len(questions)), desc = "Running Verification Evaluation"):
            benchmarker = RetrievalBenchmark(questions[i])
            score_list.append(benchmarker.answeredBy(documents[i]))
        return sum(score_list) / len(score_list)