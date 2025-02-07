import sys
sys.path.append("./")
from evaluation.benchmarker import Benchmarker
import json
import random
from retriever.embedding import EmbeddingRetriever
from retriever.reranking import RerankingRetriever
from retriever.hyde import HyDERetriever
from retriever.random import RandomRetriever
from models.huggingface import HuggingfaceModel, ModelNames
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np

CONTROL = "Control"
SCORES = "scores"
RANDOM = "Random"

def transform_data():
    """
    Scales data such that a perfect performance is scaled to the contorl
    """

    evaluation = load_data()
    control_scores = np.array(evaluation[CONTROL][SCORES])
    random_scores = np.array(evaluation[RANDOM][SCORES])
    for name in evaluation:
        clipped = np.clip(evaluation[name][SCORES], a_min= random_scores, a_max = control_scores)
        evaluation[name][SCORES] = np.divide(clipped - random_scores, control_scores - random_scores)
    del evaluation[CONTROL]
    del evaluation[RANDOM]
    return evaluation

def load_data():
    with open("combined.json", "r") as f:
        return json.load(f)

def plot(evaluation : dict):
    
    all_data = []
    labels = []
    for name, info in evaluation.items():
        labels.append(name)
        all_data.append(np.array(info["scores"]))
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    bp = ax.boxplot(
        all_data,
        vert=False,               
        showfliers=False,
        labels=labels
    )
    
    for i, box in enumerate(bp['boxes']):
        path = box.get_path()
        verts = path.vertices  
        q1 = verts[0, 0] 
        q3 = verts[2, 0] 

        median_line = bp['medians'][i]
        median_x = median_line.get_xdata().mean() 
        

        y_top = np.max(verts[:, 1])
        y_bottom = np.min(verts[:, 1])
        y_center = 0.5 * (y_top + y_bottom) + 0.1
        
        def format_percent(val : float) -> str:
            return f"{val * 100:.2f}%"

        ax.text(q1 - 0.01, y_center,
                f"Q1={format_percent(q1)}",
                ha='right',   
                va='center')
        
        ax.text(median_x + 0.025, y_center,
                f"Med={format_percent(median_x)}",
                ha='center',
                va='bottom', 
                color='red')
        
        ax.text(q3 + 0.01, y_center,
                f"Q3={format_percent(q3)}",
                ha='left',   
                va='center')
        
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_title("Retriever Performance Comparisons")
    plt.xlabel("Accuracy")
    plt.ylabel("Retrieval Method")
    plt.tight_layout()
    plt.show()

def combine():
    with open("evaluation.json", "r") as f:
        evaluation = json.load(f)
    with open("random.json", "r") as f:
        random = json.load(f)
    evaluation["Random"] = random["Random"]
    json.dump(evaluation, open("combined.json", "w"))

def run():
    # embedding = EmbeddingRetriever()
    # reranking = RerankingRetriever(embedding, first_pass_n=20)
    # hyde = HyDERetriever(HuggingfaceModel(ModelNames.llama3_2_1B.value, max_tokens= 512, temperature= 0.5))
    # rerank_hyde = RerankingRetriever(hyde, 20)
    named_retriever_dict = {
        # "Standard Vector Search": embedding,
        # "Reranked Vector Search": reranking,
        # "HyDE Vector Search": hyde,
        # "Reranked HyDE Vector Search": rerank_hyde,
        "Random" : RandomRetriever()

    }
    runner = Benchmarker(named_retriever_dict, HuggingfaceModel(ModelNames.llama3_2_1B.value, max_tokens= 100, temperature= 0), use_questions = 1000)
    evaluation = runner.eval()
    json.dump(evaluation, open("random.json", "w"))

def pass_rate():
    """
    Refactors evaluation.json into pass/fail data
    """
    thresholds = [0.5, 0.6, 0.7]
    data = load_data()
    names = list(data.keys())

    num_thresholds = len(thresholds)
    x = np.arange(len(names)) 

    bar_width = 0.1  
    offsets = np.linspace(-bar_width*(num_thresholds-1)/2, 
                        bar_width*(num_thresholds-1)/2, 
                        num_thresholds)

    for i, t in enumerate(thresholds):
        pass_rates = []
        for name in names:
            scores = np.array(data[name][SCORES])
            pass_rates.append(np.average(np.where(scores >= t, True, False)))
        bars = plt.bar(x + offsets[i], pass_rates, width=bar_width, label=f"{t*100:.0f}%")
        for bar in bars:
            y_val = bar.get_height()
            x_val = bar.get_x() + bar.get_width() / 2
            plt.text(x_val + 0.01, 
                    y_val + 0.01,         
                    f"{y_val*100:.1f}%" ,
                    ha='center',
                    va='bottom',
                    fontsize=9)

    plt.xticks(x, names) 
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xlabel('Retrieval Method')
    plt.ylabel('Pass Rate')
    plt.title('Pass Rates for Different Thresholds')
    plt.legend(title = "Threshold")
    plt.show()
    





if __name__ == "__main__":
    pass_rate()