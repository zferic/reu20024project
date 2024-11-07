import os
import re
import json
import pandas as pd

def extract_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            if file_path.endswith(".txt") and not file_path.endswith(("_1.txt", "_2.txt", "_3.txt", "_4.txt", "_5.txt")):
                print(file_path)
                text = file.read()

            
                title_match = re.search(r"Title:\s*(.*?)\s*(?=Authors:|Publication Date:|###|$)", text, re.DOTALL)
                authors_match = re.search(r"Authors:\s*(.*?)\s*(?=Publication Date:|###|$)", text, re.DOTALL)
                publication_date_match = re.search(r"Publication Date:\s*(.*?)\s*(?=###|$)", text, re.DOTALL)
                abstract_match = re.search(r"### Abstract ###\s*(.*?)\s*(?=###|$)", text, re.DOTALL)
                introduction_match = re.search(r"### Introduction ###\s*(.*?)\s*(?=###|$)", text, re.DOTALL)
                results_match = re.search(r"### Results ###\s*(.*?)\s*(?=###|$)", text, re.DOTALL)
                discussion_match = re.search(r"### Discussion ###\s*(.*?)\s*(?=###|$)", text, re.DOTALL)
                conclusion_match = re.search(r"### Conclusion ###\s*(.*?)\s*(?=###|$)", text, re.DOTALL)

            
                title = title_match.group(1).strip() if title_match else ""
                authors = authors_match.group(1).strip() if authors_match else ""
                publication_date = publication_date_match.group(1).strip() if publication_date_match else ""
                abstract = abstract_match.group(1).strip() if abstract_match else ""
                introduction = introduction_match.group(1).strip() if introduction_match else ""
                results = results_match.group(1).strip() if results_match else ""
                discussion = discussion_match.group(1).strip() if discussion_match else ""
                conclusion = conclusion_match.group(1).strip() if conclusion_match else ""

                return {
                    "title": title,
                    "authors": authors,
                    "publication_date": publication_date,
                    "abstract": abstract,
                    "introduction": introduction,
                    "results": results,
                    "discussion": discussion,
                    "conclusion": conclusion
                }

            elif file_path.endswith(".json"):
                data = json.load(file)
                return data  

    except Exception as e:
        print(f"Error extracting text from file {file_path}: {e}")
        return None

def extract_texts_from_directory(directory):
    texts = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".txt") and not file_name.endswith(("_1.txt", "_2.txt", "_3.txt", "_4.txt", "_5.txt")):
            file_path = os.path.join(directory, file_name)
            text = extract_text(file_path)
            print(file_name)
            print(text)
            if text:
                if isinstance(text, list):
                    texts.extend(text)
                elif isinstance(text, dict):
                    texts.append(text)
                else:
                    print(f"Unsupported data type in file {file_path}: {type(text)}")
    return texts

if __name__ == "__main__":
    directory = r"/media/zman/extrahd/reu20024project/papers/papers"  
    texts = extract_texts_from_directory(directory)

    for i, item in enumerate(texts):
        
        if not isinstance(item, dict):
            print(f"Item at index {i} is not a dictionary: {item}")

    df = pd.DataFrame(texts)
    df.to_csv("final_texts.csv", index=False)
    print(f"Saved {len(texts)} texts to 'final_texts.csv'")
