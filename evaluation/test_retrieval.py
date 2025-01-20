import unittest
import sys
sys.path.append("./")
from evaluation.retrieval import RetrievalBenchmark

class TestRetrieval(unittest.TestCase):

    def test_basic(self):
        """
        Showcases basic functionality works
        """
        prompt = RetrievalBenchmark("What is the capital of France?")
        self.assertGreater(prompt.answeredBy("Paris is in France"), 0.99)
        self.assertLess(prompt.answeredBy("France is a country in Europe."), 0.01)
    

    def test_protect(self):
        """
        Showcases functionality using Q&A's from relevant papers.
        """
        prompt = RetrievalBenchmark("Was there a significant association between M 1 -dG and air-formaldehyde in non-smokers?")
        self.assertGreater(prompt.answeredBy("When the exposed workers and controls were subgrouped according to smoking, M 1 -dG tended to increase in all the subjects but a significant association between M 1 -dG and air-formaldehyde was only found in not smokers (p = 0.009)."), 0.90)
        self.assertLess(prompt.answeredBy("Individuals who reported always using sunscreen had significantly higher urinary concentrations of triclosan, methyl, ethyl, and propyl parabens, and BP3 (59, 92, 102, 151, and 510% higher, respectively) compared to \u201cNever\u201d users of sunscreen."), 0.10)
        self.assertLess(prompt.answeredBy("Associations with mouthwash use were generally stronger in men compared to women"), 0.10)
        self.assertLess(prompt.answeredBy("Was there a significant association between M 1 -dG and air-formaldehyde in non-smokers?"), 0.10) # This test is more difficult for the model


if __name__ == "__main__":
    unittest.main()