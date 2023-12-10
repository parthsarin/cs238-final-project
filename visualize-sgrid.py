from policy.DeepQMemoryless.Student import StudentQ, StudentPolicy, StudentObservation
import matplotlib.pyplot as plt
import seaborn as sns
import torch

sns.set_theme()


sQ = StudentQ(32)
sQ.load_state_dict(torch.load('model/deep-q-memoryless-student.pt'))
sπ = StudentPolicy(sQ)

grid = [[None for _ in range(8)] for _ in range(15)]
for na in range(15):
    for ft in range(8):
        o = StudentObservation(None, ft, na)
        a = sπ.action(o)
        if a.submit:
            grid[na][ft] = 0
        else:
            grid[na][ft] = a.work

ax = sns.heatmap(grid, annot=True)
ax.invert_yaxis()

plt.title("Memoryless DQN policy")
plt.xlabel("free time (hours)")
plt.ylabel("num assignments")
plt.show()
