import unittest
from benchmark import CorpusBenchmarker

class TestBenchmark(unittest.TestCase):

    def test_basic(self):
        """
        Showcases basic functionality works
        """
        corpus = CorpusBenchmarker(["The Capitol of France is Pairs"])
        self.assertGreater(corpus.supports("Paris is in France"), 0.99)
        self.assertLess(corpus.supports("Paris is in the United Kingdom"), 0.01)

    def test_transitive(self):
        """
        Making inferences from connected seperate documents would be desireable, but this currently is not implemented
        """
        corpus = CorpusBenchmarker(["The table is red", "The ball is the same color as the table"])
        # self.assertGreater(corpus.supports("The ball is red"), 0.99)

    def test_fictious(self):
        """
        Highlights that the CorpusBenchmarker is not limited to verifiying known information
        """
        corpus = CorpusBenchmarker(["Two plus two is 5"])
        self.assertGreater(corpus.supports("2 + 2 = 5"), 0.99)
        self.assertLess(corpus.supports("2 + 2 = 4"), 0.01)
    


if __name__ == "__main__":
    unittest.main()