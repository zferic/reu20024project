from openai import OpenAI
import openai

from openai import OpenAI


client = OpenAI(api_key = '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'
)

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Write a haiku about recursion in programming."
        }
    ]
) 

print(completion.choices[0].message)



import os
import re

# Define the directory containing the text files
text_files_dir = "/media/zman/extrahd/reu20024project/preprocessing/docs"

# Function to extract sections from a paper
def extract_sections(text):
    abstract_match = re.search(r'### Abstract ###(.*?)### Introduction ###', text, re.DOTALL)
    introduction_match = re.search(r'### Introduction ###(.*?)(###|$)', text, re.DOTALL)
    conclusion_match = re.search(r'### Conclusion ###(.*?)(###|$)', text, re.DOTALL)
    
    abstract = abstract_match.group(1).strip() if abstract_match else ""
    introduction = introduction_match.group(1).strip() if introduction_match else ""
    conclusion = conclusion_match.group(1).strip() if conclusion_match else ""
    
    return abstract, introduction, conclusion

# Initialize lists to store the extracted sections
abstracts = []
introductions = []
conclusions = []

# Read all text files and extract sections
for filename in os.listdir(text_files_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(text_files_dir, filename), 'r', encoding='utf-8') as file:
            text = file.read()
            abstract, introduction, conclusion = extract_sections(text)
            if len(abstract.strip()) > 100:
                abstract = abstract.replace('\n', ' ').strip()
                #print("Abstract", abstract.strip())
                abstracts.append(abstract)
            if len(introduction.strip()) > 100:
                introduction = introduction.replace('\n', ' ').strip()
                #print("Introduction", introduction.strip())
                introductions.append(introduction)
            if len(conclusion.strip()) > 100:
                conclusion = conclusion.replace('\n', ' ').strip()
                #print("Conclusion", conclusion.strip())
                conclusions.append(conclusion)
# Load the dataset
global_results = []

for abstract in abstracts:

    print(abstract)

    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are good at making question-answer pairs from unstructured data."},
        {
            "role": "user",
            "content": "Give me 10 question-answer-context data points from the following text and give a json format: " + abstract
        }
    ]
    )   

    global_results.append(completion.choices[0].message.content)


    import json

# Initialize an empty list to hold the concatenated results
# Initialize an empty list to hold the concatenated results
# Initialize an empty list to hold the concatenated results
concatenated_json = []

# Loop over each entry in global_results
for index, result in enumerate(global_results):
    try:
        # Remove the ```json and ``` markers at the start and end
        result_cleaned = result.replace('```json', '').replace('```', '').strip()
        print(result_cleaned)
        # Parse the cleaned JSON string
        parsed_result = json.loads(result_cleaned)
        
        # Extend the concatenated_json list with the parsed result
        concatenated_json.extend(parsed_result)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON at index {index}: {e}")
        continue  # Skip the current entry if it's malformed

# Write the final concatenated list to a file named 'outputs_final.json'
if concatenated_json:
    with open('outputs_final.json', 'w') as file:
        json.dump(concatenated_json, file, indent=4)
    print("Concatenated JSON has been written to outputs_final.json")
else:
    print("No valid JSON data was loaded.")