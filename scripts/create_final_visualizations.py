
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load evaluation results
with open('data/evaluation_results.json', 'r') as f:
    results = json.load(f)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. Accuracy Comparison Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))

pipelines = list(results.keys())
accuracies = [results[p]['accuracy'] * 100 for p in pipelines]
f1_scores = [results[p]['macro_f1'] * 100 for p in pipelines]

x = np.arange(len(pipelines))
width = 0.35

bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
bars2 = ax.bar(x + width/2, f1_scores, width, label='Macro F1', alpha=0.8)

ax.set_xlabel('Pipeline', fontsize=12)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Pipeline Performance Comparison (Gold Standard Evaluation)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(pipelines)
ax.legend()
ax.set_ylim(0, 100)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.savefig('data/plots/gold_standard_accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Confusion Matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, (name, metrics) in enumerate(results.items()):
    if idx < 4:  # Maximum 4 pipelines
        cm = np.array(metrics['confusion_matrix'])

        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=['Positive', 'Neutral', 'Negative'],
                    yticklabels=['Positive', 'Neutral', 'Negative'],
                    ax=axes[idx])
        axes[idx].set_title(f'{name} Pipeline')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')

plt.suptitle('Confusion Matrices (Normalized)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('data/plots/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()

print("[DONE] Visualizations created in data/plots/")
