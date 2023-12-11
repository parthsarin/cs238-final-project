from policy.DeepQMemoryless.Student import StudentQ, StudentPolicy, StudentObservation
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

sns.set_theme()


sQ = StudentQ(32)
sQ.load_state_dict(torch.load('model/deep-q-memoryless-student.pt'))
sπ = StudentPolicy(sQ)

grid = np.zeros((10, 7))
for na in range(10):
    for ft in range(1, 8):
        o = StudentObservation(None, ft, na)
        a = sπ.action(o)
        if a.submit:
            grid[na][ft - 1] = -1
        else:
            grid[na][ft - 1] = a.work

ax = sns.heatmap(grid, mask=grid < 0, annot=True, cbar_kws={
                 'label': r'% of time working'})
ax.invert_yaxis()
ax.set_xticklabels(list(range(1, 8)))

cmap1 = mpl.colors.ListedColormap(['c'])
ax = sns.heatmap(grid, mask=grid >= 0, cmap=cmap1, annot=False, cbar=False)
ax.invert_yaxis()
ax.set_xticklabels(list(range(1, 8)))

plt.title("Memoryless DQN policy")
plt.xlabel("free time (hours)")
plt.ylabel("num assignments")
plt.show()
