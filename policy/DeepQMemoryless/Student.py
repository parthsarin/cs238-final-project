from env.POMDP import MemorylessPolicy
from env.Student import StudentAction, StudentObservation, Student
from policy.RandomPolicy import rand_action

from random import random as rand

import torch
import torch.nn as nn
import torch.nn.functional as F


A = [StudentAction(submit=True)] + [
    StudentAction(False, r, 1 - r)
    for r in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
]


class StudentQ(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, len(A))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class StudentPolicy(MemorylessPolicy):
    def __init__(self, q: StudentQ, eps: float = 0.1):
        self.q = q
        self.eps = eps

    @staticmethod
    def to_tensor(o: StudentObservation):
        inp = [o.assignment_grade, o.free_time, o.num_assignments]
        if o.assignment_grade is None:
            inp[0] = -1

        return torch.tensor(inp, dtype=torch.float32)

    def action(self, o: StudentObservation):
        # with some probability, take a random action
        if rand() < self.eps:
            return rand_action(o, StudentAction)

        # otherwise, take the best action
        q_vals = self.q(self.to_tensor(o))
        for idx in q_vals.argsort(descending=True):
            a = A[idx.item()]
            if StudentAction.is_valid(o, a):
                return a

        return rand_action(o, StudentAction)

    def loss(
        self,
        o: StudentObservation,
        a: StudentAction,
        r: float,
        op: StudentObservation
    ):
        try:
            a_idx = A.index(a)
        except ValueError:
            return 0

        # predicted q value
        q_vals = self.q(self.to_tensor(o))
        q = q_vals[a_idx]

        # target q value
        q_target = r + 0.95 * self.q(self.to_tensor(op)).max()

        return F.mse_loss(q, q_target)
