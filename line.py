import matplotlib.pyplot as plt
import numpy as np

plt.rc('font', family='Times New Roman')
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), dpi=100, tight_layout=True)
x_axis_data = [-3, -2, -1, 0, 1, 2, 3]
y_axis_data1 = [0.821, 0.849, 0.839, 0.892, 0.921, 0.873, 0.861]
y_axis_data2 = [0.932, 0.93, 0.934, 0.94, 0.95, 0.904, 0.894]
y_axis_data3 = [0.912, 0.918, 0.925, 0.919, 0.919, 0.918, 0.917]
y_axis_data4 = [0.935, 0.898, 0.926, 0.942, 0.950, 0.941, 0.906]

axs[0].plot(x_axis_data, y_axis_data1, 'b*--', alpha=0.7, linewidth=2, label='in silico')
axs[0].plot(x_axis_data, y_axis_data2, 'rs--', alpha=0.7, linewidth=2, label='S. aureus')
axs[0].plot(x_axis_data, y_axis_data3, 'go--', alpha=0.7, linewidth=2, label='E. coli')
axs[0].plot(x_axis_data, y_axis_data4, 'm+--', alpha=0.7, linewidth=2, label='S. cerevisiae')
axs[0].legend()
axs[0].set_xlabel('$\lambda$')
axs[0].set_ylabel('AUROC')
labels = axs[0].get_xticklabels() + axs[0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
axs[0].grid(axis='y')

y_axis_data1 = [0.788, 0.823, 0.814, 0.862, 0.900, 0.848, 0.833]
y_axis_data2 = [0.930, 0.905, 0.892, 0.917, 0.944, 0.883, 0.866]
y_axis_data3 = [0.911, 0.910, 0.917, 0.911, 0.909, 0.908, 0.906]
y_axis_data4 = [0.914, 0.865, 0.892, 0.915, 0.940, 0.935, 0.900]

axs[1].plot(x_axis_data, y_axis_data1, 'b*--', alpha=0.7, linewidth=2, label='in silico')
axs[1].plot(x_axis_data, y_axis_data2, 'rs--', alpha=0.7, linewidth=2, label='S. aureus')
axs[1].plot(x_axis_data, y_axis_data3, 'go--', alpha=0.7, linewidth=2, label='E. coli')
axs[1].plot(x_axis_data, y_axis_data4, 'm+--', alpha=0.7, linewidth=2, label='S. cerevisiae')
axs[1].legend()
axs[1].set_xlabel('$\lambda$')
axs[1].set_ylabel('AUPRC')
labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
axs[1].grid(axis='y')
plt.savefig('fig3.pdf', dpi=300)
plt.show()
