import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV
df = pd.read_csv("results.csv")

# Set a clean plot style
sns.set(style="whitegrid")

# Sort by accuracy (optional, for better visual effect)
df_sorted = df.sort_values(by="accuracy", ascending=False)

# Create a horizontal barplot
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=df_sorted, x="accuracy", y="experiment", palette="viridis")

# Increase font size of y-axis labels (experiment names)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)  # Puedes ajustar el número según lo que necesites


# Add accuracy labels to the bars
for i, (accuracy, experiment) in enumerate(zip(df_sorted["accuracy"], df_sorted["experiment"])):
    ax.text(accuracy + 0.001, i, f"{accuracy:.3f}", va='center', fontsize = 16)

plt.title("precisión por Experimento", fontsize=18)
plt.xlabel("precisión", fontsize=16)
plt.ylabel("Experimento", fontsize=16)
plt.xlim(0.85, 0.95)  # Adjust limits for better visual comparison
plt.tight_layout()
plt.savefig("results.pdf")
plt.show()
