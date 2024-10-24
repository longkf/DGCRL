import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.rc('font', family='Times New Roman')

data1 = np.array([
    [0.95, 0.95, 0.93, 0.93, 0.93],
    [0.95, 0.93, 0.92, 0.94, 0.93],
    [0.94, 0.91, 0.93, 0.94, 0.92],
    [0.85, 0.88, 0.92, 0.92, 0.90],
    [0.84, 0.88, 0.92, 0.90, 0.92]
])
data2 = np.array([
    [0.94, 0.92, 0.89, 0.89, 0.88],
    [0.94, 0.91, 0.89, 0.91, 0.89],
    [0.94, 0.89, 0.90, 0.90, 0.89],
    [0.83, 0.87, 0.90, 0.90, 0.89],
    [0.82, 0.86, 0.90, 0.88, 0.90]
])
x_labels = ['16', '32', '64', '128', '256']
y_labels = ['256', '128', '64', '32', '16']

# 创建图形和轴
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

axs[0] = sns.heatmap(data1, annot=True, cmap="OrRd", xticklabels=x_labels, yticklabels=y_labels, ax=axs[0])
axs[0].set_xlabel('output channels')
axs[0].set_ylabel('hidden channels')
axs[0].set_title('AUROC')
labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

axs[1] = sns.heatmap(data2, annot=True, cmap="OrRd", xticklabels=x_labels, yticklabels=y_labels, ax=axs[1])
axs[1].set_xlabel('output channels')
axs[1].set_ylabel('hidden channels')
axs[1].set_title('AUPRC')
labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

plt.savefig('fig4.pdf', dpi=300)
plt.show()
