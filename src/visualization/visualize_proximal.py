import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ['NMMR-V', 'NMMR-V', 'NMMR-U', 'NMMR-U']
models = ['MLP', 'Ours', 'MLP', 'Ours']
sizes = ['1000', '5000', '10000', '50000']

data = np.array([
    [23.41, 30.74, 42.88, 62.18],
    [21.54, 24.46, 21.37, 27.50],
    [23.68, 16.21, 14.25, 14.27],
    [10.69, 7.67, 5.56, 6.51]
])

# IQR data
iqr = np.array([
    [11.26, 17.73, 29.45, 16.97],
    [17.42, 17.93, 10.12, 16.30],
    [8.02, 10.55, 4.46, 12.47],
    [14.72, 6.70, 6.72, 5.90]
])

# Colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Create the plot
fig, ax = plt.subplots(figsize=(14, 6))

bar_width = 0.2
index = np.arange(len(sizes))

for i in range(len(methods)):
    bars = ax.bar(index + i * bar_width, data[i], bar_width,
                  label=f'{methods[i]} ({models[i]})', color=colors[i], alpha=0.8)

    # Add error bars
    ax.errorbar(index + i * bar_width, data[i], yerr=iqr[i] / 2, fmt='none', color='black', capsize=5)

    # Add value labels
    for j, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{data[i][j]:.2f}',
                ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Training Set Size')
ax.set_ylabel('Median c-MSE')
#ax.set_title('Median c-MSE for Demand Dataset Across Different Training Set Sizes')
ax.set_xticks(index + bar_width * 1.5)
ax.set_xticklabels(sizes, fontsize=12)
ax.legend()

plt.tight_layout()
plt.savefig(f'experiments/results/proximal/proximal.png')