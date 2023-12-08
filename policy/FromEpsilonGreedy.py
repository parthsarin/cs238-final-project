import random
from env import MemorylessPolicy, Observation, Action
from collections import defaultdict
from sklearn.neighbors import KDTree
from pickle import load
from .RandomPolicy import rand_action


class EpsilonGreedyPolicy(MemorylessPolicy):
    def __init__(
        self,
        Q,
        obs_ordering: list,
        act_ordering: list,
        act_class: type(Action),
        q_thresh: float = 1e2,
        is_valid: callable = lambda o, a: True,
        epsilon: float = 0.1
    ):
        self.Q = Q

        # extract the states and actions from the Q table
        stored_actions = defaultdict(list)
        stored_obs = []
        for (o, a) in Q:
            stored_actions[o].append(a)
            stored_obs.append(o)

        # build a KDTree for the states
        self.tree = KDTree(stored_obs)
        self.stored_actions = stored_actions
        self.q_thresh = q_thresh

        # store the orderings to turn the observations and actions into tuples
        self.obs_ordering = obs_ordering
        self.act_ordering = act_ordering
        self.act_class = act_class
        self.is_valid = is_valid

        # store statistics
        self.num_rand = 0
        self.num_q = 0
        self.epsilon = epsilon

    def action(self, o: Observation):
        # get the observation as a tuple
        o_raw = o
        o = tuple(getattr(o, field) for field in self.obs_ordering)
        o = tuple(v if v is not None else 1e9 for v in o)

        # get a random number between 0 and 1, and if it's less than epsilon, take a random action
        if random.random() < self.epsilon:
            self.num_rand += 1
            return rand_action(o_raw, self.act_class)

        # get the nearest neighbor
        dist, idx = self.tree.query([o], k=1)
        o_nbr = tuple(self.tree.data[idx[0][0]])

        # if we're close enough, take the action with the highest Q value
        if dist[0][0] < self.q_thresh:
            possible_actions = self.stored_actions[o_nbr]

            # get the action with the highest Q value
            possible_actions = sorted(
                possible_actions,
                key=lambda a: self.Q[(o_nbr, a)],
                reverse=True
            )
            # go in order until we find a valid action
            for a in possible_actions:
                a = tuple(v if v != 1e9 else None for v in a)
                a = self.act_class(**dict(zip(self.act_ordering, a)))

                if not a.is_valid(o_raw, a):
                    continue

                self.num_q += 1
                return a

        # otherwise, or if there are no valid actions, take a random action
        self.num_rand += 1
        return rand_action(o_raw, self.act_class)


def load_epsilon_greedy_policy(filename: str, act_class: type(Action)):
    data = load(open(filename, "rb"))
    return EpsilonGreedyPolicy(
        data["Q"],
        data["obs_ordering"],
        data["act_ordering"],
        act_class
    )
