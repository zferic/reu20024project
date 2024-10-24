# Transform the data into the desired format

import json

# Define the input and output file paths
input_file = "outputs_final_introduction.json"
output_file = "training_qa_instruct_format_intro.json"

# Read the original QA dataset
with open(input_file, "r") as f:
    qa_data = json.load(f)

transformed_data = []
for item in qa_data:
    transformed_data.append({
        "text": f"<human>: {item['question']}\n<bot>: {item['answer']}",
        "metadata": {"source": "unified_chip2"}
    })

with open("output_file.jsonl", "w") as f:
    for entry in transformed_data:
        f.write(json.dumps(entry) + '\n')

