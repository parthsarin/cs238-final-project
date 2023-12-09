from math import log, sqrt
from numpy import argmax
from env import MemorylessPolicy, Observation, Action
from collections import defaultdict
from sklearn.neighbors import KDTree
from pickle import load
from .RandomPolicy import rand_action


class UCB1Policy(MemorylessPolicy):
    def __init__(
        self,
        Q,
        obs_ordering: list,
        act_ordering: list,
        act_class: type(Action),
        is_valid: callable = lambda o, a: True,
        o_counter: defaultdict = None,
        o_a_counter: defaultdict = None,
        c: int = 2,
    ):
        self.Q = Q
        self.c = c
        if o_counter is None:
            o_counter = defaultdict(int)
        
        if o_a_counter is None:
            o_a_counter = defaultdict(int)
        
        self.o_counter = o_counter
        self.o_a_counter = o_a_counter

        # extract the states and actions from the Q table
        stored_actions = defaultdict(list)
        stored_obs = []
        for (o, a) in Q:
            stored_actions[o].append(a)
            stored_obs.append(o)

        self.stored_actions = stored_actions

        # store the orderings to turn the observations and actions into tuples
        self.obs_ordering = obs_ordering
        self.act_ordering = act_ordering
        self.act_class = act_class
        self.is_valid = is_valid

        # store statistics
        self.num_rand = 0
        self.num_q = 0

    def action(self, o: Observation):
        o_raw = o
        o_list = []
        for field in self.obs_ordering:
            v = getattr(o, field)
            if field == "assignment_grade" and v is not None:
                v = round(v)
            o_list.append(v)
        o = tuple(o_list)
        o = tuple(v if v is not None else 1e9 for v in o)
        self.o_counter[o] += 1
        action_space = self.stored_actions[o]
        next_action = {}
        for a in action_space:
            if self.o_a_counter[(o, a)] == 0:
                b = float('inf')
            else:
                b = self.Q[(o, a)] + self.c * sqrt(log(self.o_counter[o]) / self.o_a_counter[(o, a)])
            next_action[a] = b
        # if there are no valid actions in action space, return random action
        if len(next_action) == 0:
            self.num_rand += 1
            return rand_action(o_raw, self.act_class)
        a_max = max(next_action, key=next_action.get)
        self.o_a_counter[(o, a_max)] += 1
        a_max = tuple(v if v != 1e9 else None for v in a_max)
        a_max = self.act_class(**dict(zip(self.act_ordering, a_max)))
        return a_max


def load_ucb1_policy(filename: str, act_class: type(Action)):
    data = load(open(filename, "rb"))
    return UCB1Policy(
        data["Q"],
        data["obs_ordering"],
        data["act_ordering"],
        act_class
    )
