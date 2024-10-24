import json

# Define the input and output file paths
input_file = "training_qa.json"
output_file = "training_qa_instruct_format.json"

# Function to convert QA pairs to instruction format
def convert_qa_to_instruction_format(qa_data):
    instruction_format_data = []
    
    for entry in qa_data:
        question = entry.get("question", "")
        context = entry.get("context", "")
        answer = entry.get("answer", "")
        
        # Create the instruction with the question
        instruction = f"Answer the following question based on the given context: {context}"
        
        # Append the instruction, context as input, and the answer as output
        instruction_format_data.append({
            "instruction": instruction,
            "input": question,
            "output": answer
        })
    
    return instruction_format_data

# Read the original QA dataset
with open(input_file, "r") as f:
    qa_data = json.load(f)

# Convert the QA dataset to instruction format
instruction_data = convert_qa_to_instruction_format(qa_data)

# Write the converted data to a new JSON file
with open(output_file, "w") as f:
    json.dump(instruction_data, f, indent=4)

print(f"Converted dataset saved to {output_file}")
