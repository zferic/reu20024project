import matplotlib.pyplot as plt

# Updated Data
models = ['RAG Model', 'ChatGPT', 'Meta AI', 'Gemini', 'GPT Turbo']
correct_responses = [23, 20, 9, 2, 0]
accuracy = [92, 80, 36, 8, 0]

# Creating the figure and axes
fig, ax1 = plt.subplots()

# Bar plot for correct responses
color = 'tab:red'
ax1.set_xlabel('Models')
ax1.set_ylabel('Correct Responses', color=color)
ax1.bar(models, correct_responses, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Twin axes for accuracy
ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Accuracy (%)', color=color)
ax2.plot(models, accuracy, color=color, marker='o', linestyle='--', linewidth=2, markersize=8)
ax2.tick_params(axis='y', labelcolor=color)

# Title and layout adjustments
plt.title('Model Performance: Correct Responses and Accuracy')
fig.tight_layout()

# Display the plot
plt.show()

