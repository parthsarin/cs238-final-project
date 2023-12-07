from random import random as rand
from policy.RandomPolicy import rand_action
from env import TeacherObservation, TeacherAction, Policy

import torch
import torch.nn as nn
import torch.nn.functional as F


A = [
    TeacherAction(r, g, pd)
    for r in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
    for g in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
    for pd in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
    if r + g + pd == 1
]


class TeacherQ(nn.Module):
    def __init__(self, history_dim, hidden_dim):
        super().__init__()
        self.history_dim = history_dim
        self.hidden_dim = hidden_dim

        # add the history encoder
        self.rnn = nn.GRU(5, history_dim)

        # add the layers
        self.fc1 = nn.Linear(history_dim + 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, len(A))

    @staticmethod
    def o_to_tensor(o: TeacherObservation):
        inp = [o.free_time, o.num_assignments]
        return torch.tensor(inp, dtype=torch.float32)

    @staticmethod
    def a_to_tensor(a: TeacherAction):
        inp = [a.rest, a.grading, a.pd]
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


class TeacherPolicy(Policy):
    def __init__(self, q: TeacherQ, eps: float = 0.1, train: bool = True):
        self.q = q
        self.eps = eps
        self.train = train

    def action(self, history):
        # with some probability, take a random action
        if rand() < self.eps:
            if self.train:
                return rand_action(history[-1][0], TeacherAction), None
            else:
                return rand_action(history[-1][0], TeacherAction)

        # otherwise, take the best action
        q_vals = self.q(history)
        for idx in q_vals.argsort(descending=True):
            a = A[idx.item()]
            if TeacherAction.is_valid(history[-1][0], a):
                if self.train:
                    return a, q_vals
                else:
                    return a

        if self.train:
            return rand_action(history[-1][0], TeacherAction), None
        else:
            return rand_action(history[-1][0], TeacherAction)

    def loss(
        self,
        q_vals,
        a: TeacherAction,
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
