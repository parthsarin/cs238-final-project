from random import random as rand
from random import choice
from math import sqrt, log
from collections import defaultdict
from env import StudentObservation, StudentAction, Policy

import torch
import torch.nn as nn
import torch.nn.functional as F


A = [StudentAction(submit=True)] + [
    StudentAction(False, r, 1 - r)
    for r in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
]


class StudentQ(nn.Module):
    def __init__(self, history_dim, hidden_dim):
        super().__init__()
        self.history_dim = history_dim
        self.hidden_dim = hidden_dim

        # add the history encoder
        self.rnn = nn.GRU(6, history_dim)

        # add the layers
        self.fc1 = nn.Linear(history_dim + 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, len(A))

    @staticmethod
    def o_to_tensor(o: StudentObservation):
        inp = [o.assignment_grade, o.free_time, o.num_assignments]
        if o.assignment_grade is None:
            inp[0] = -1
        return torch.tensor(inp, dtype=torch.float32)

    @staticmethod
    def a_to_tensor(a: StudentAction):
        if a.submit:
            return torch.tensor([1, 0, 0], dtype=torch.float32)

        inp = [0, a.rest, a.work]
        return torch.tensor(inp, dtype=torch.float32)

    def forward(self, history):
        h = torch.zeros(self.history_dim)
        for (o, a) in history:
            if a is None:
                break

            # encode the observation and action
            o_tensor = self.o_to_tensor(o)
            a_tensor = self.a_to_tensor(a)

            # concatenate the observation and action
            x = torch.cat([o_tensor, a_tensor])

            # h = F.tanh(self.Wh @ h + self.Wx @ x + self.bh)
            _, h = self.rnn(x.view(1, 1, -1), h.view(1, 1, -1))

        h = h.view(-1)

        # get the last observation
        o = history[-1][0]
        o_tensor = self.o_to_tensor(o)

        # concatenate the history and observation
        inp = torch.cat([h, o_tensor])

        # run the layers
        x = F.relu(self.fc1(inp))
        x = self.fc2(x)
        return x


class StudentPolicy(Policy):
    def __init__(self, q: StudentQ, eps: float = 1e-4, train: bool = True, c: float = 1.0):
        self.q = q
        self.eps = eps
        self.train = train

        # N(o, a) = number of times we've seen (o, a)
        self.N = defaultdict(int)
        self.c = c

    def rand_action(self, o: StudentObservation):
        while True:
            a = choice(A)
            if StudentAction.is_valid(o, a):
                return a

    def a_inference(self, history):
        """
        The action during inference time (no backprop)
        """
        q_vals = self.q(history)
        for idx in q_vals.argsort(descending=True):
            a = A[idx.item()]
            if StudentAction.is_valid(history[-1][0], a):
                return a

        return self.rand_action(history[-1][0])

    def action(self, history):
        if not self.train:
            return self.a_inference(history)

        # calculate the q_vals for backprop
        q_vals = self.q(history)
        a_score = q_vals.clone()

        # add on UCB1 heuristic
        o = history[-1][0]
        N = sum(self.N[(o, a)] for a in range(len(A))) + self.eps
        for idx in range(len(A)):
            if self.N[(o, idx)] == 0:
                a_score[idx] = float('inf')
            else:
                a_score[idx] += self.c * sqrt(log(N) / self.N[(o, idx)])

        # take the best action
        for idx in a_score.argsort(descending=True):
            a = A[idx.item()]
            if StudentAction.is_valid(o, a):
                return a, q_vals

        return self.rand_action(o)

    def loss(
        self,
        q_vals,
        a: StudentAction,
        r: float,
        new_history,
    ):
        # get the actoin that was taken
        try:
            a_idx = A.index(a)
        except ValueError:
            return 0

        # get the q value that was predicted
        q = q_vals[a_idx]

        # get the target
        q_target = r + 0.95 * self.q(new_history).max()

        # compute the loss
        return F.mse_loss(q, q_target)
