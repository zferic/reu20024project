import sys
sys.path.append("./")
from models.abstract import AbstractModel
from utils.messages import MessageHistory
from langchain_core.documents import Document


class PromptRetrievalJudge:
    """
    Allows for the benchmarking of retrieval through checks that retrieved documents contain the answer to a given question. This class is constructed using the prompt to be answered.

    RetrievalBenchmark("What is 2 + 2").answeredBy(["Two plus two is 4"]) > 0.99

    RetrievalBenchmark("What indicates that the molecular genotoxicity assay is suitable for detection")
        .answeredBy("The proposed molecular endpoints derived from the toxicogenomics assays, namely TELI 
                    (Transcriptional Effect Level Index) and PELI (Protein Effect Level Index), correlated well with the phenotypic DNA damage endpoints from comet tests,
                      suggesting that the molecular genotoxicity assay is suitable for genotoxicity detection.") > 0.99
    """

    def __init__(self, prompt : str, model : AbstractModel):
        self.prompt = prompt
        self.model = model

    def _answeredBySingle(self, document : str) -> float:
        """
        Returns a probability, from 0 to 1, that the given document is needed to answer the user's prompt.
        """

        SYS_PROMPT = """I want you to help benchmark my Retrieval Augmented Generation pipeline. I'm going to send you a user's prompt, or question, with a retrieved document that should contain information needed
to provide a user a response. If the given document is in fact needed, reply with the word 'true'. If it is not, reply with 'false'. You should not consider whether the provided prompt can be answered by common knowledge or not in your evaluation,
only consider whether the document has the relevant information or not. Pay attention if the given document is relevant to the prompt but does not actually provide an answer to the user's question, this should be marked as 'false'.""" 

        def make_user_message(given_doc : str, given_question : str):
            return f"Here is the document:\n{given_doc}\n\nHere is the question:\n{given_question}."


        EXAMPLES = [
            # [document, question, answer]
            ["Exclusion criteria included: use of tobacco products other than cigarettes (e.g., cigars, e-cigarettes, chewing tobacco), the presence of chronic liver or kidney disease, as these conditions could affect toxin metabolism, and pregnancy or breastfeeding", 
             "What exclusion criteria were considered for the study?",
             "true"],
             
             ["Participants were classified as current smokers, former smokers, or never-smokers based on self-report. Current smoking status was confirmed by measurement of plasma cotinine levels (>10 ng/mL). Pack-years were computed as (packs smoked per day) × (years of smoking).",
              "Does the metabolite cotinine have any psychoactive effect in smokers?",
              "false"],

             ["* Chen C, Wang X, Wang L, et al.  Effect of environmental tobacco smoke on levels of urinary hormone markers. Environ Health Perspect. 2005;113(4):412–417. doi: 10.1289/ehp.7436. [ DOI ] [ PMC free article ] [ PubMed ] [ Google Scholar ] (2005).",
              "What is the effect of environmental tobacco smoke on levels of urinary hormone markers?",
              "false"]
        ]


        messages = MessageHistory(SYS_PROMPT)
        for ex in EXAMPLES:
            messages.add_user_message(make_user_message(ex[0], ex[1]))
            messages.add_model_message(ex[2])
        messages.add_user_message(make_user_message(document, self.prompt))
        return self._true_probability(messages)
    
    def _true_probability(self, messages : MessageHistory) -> float:
        prob_dict = self.model.next_probabilities(messages)
        true_sum = 0
        for token in prob_dict:
            if "true" in token.lower():
                true_sum += prob_dict[token]
        return true_sum

    def answeredBy(self, documents : list[Document]) -> float:
        prob_list = []
        for doc in documents:
            prob_list.append(self._answeredBySingle(doc.page_content))
        return sum(prob_list) / len(prob_list)



        


