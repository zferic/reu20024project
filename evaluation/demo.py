from runner import BenchmarkRunner, QA_PATH
import json
import random

"""
This file demonstrates how to use the Benchmark runner, using a mock retriever.
"""

def dummy_retriever(prompt : str) -> list[str]:
    """
    Retrives a random context document from the q&a json for any prompt. 
    """
    CONTEXT = "context"
    with open(QA_PATH, "r") as f:
        data = json.load(f)
    selected = [d[CONTEXT] for d in random.choices(data)]
    return selected


if __name__ == "__main__":
    runner = BenchmarkRunner(dummy_retriever, use_questions = 10)
    evaluation = runner.eval()
    print(f"BenchmarkRunner evaluated dummy_retriever's performance as {evaluation:.10f}")