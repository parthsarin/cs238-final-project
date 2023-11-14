import numpy as np
from typing import List

# ------------------------------------------------------------------------------
# generic classes for POMDPs
# ------------------------------------------------------------------------------
class State:
    def __hash__(self):
        raise NotImplementedError("hash not implemented")

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError("eq not implemented")


class Action:
    def __hash__(self):
        raise NotImplementedError("hash not implemented")

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError("eq not implemented")


class Observation:
    def __hash__(self):
        raise NotImplementedError("hash not implemented")

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError("eq not implemented")


class Policy:
    def action(self, os: List[Observation]):
        raise NotImplementedError("action not implemented")

    def __getitem__(self, os: List[Observation]):
        return self.action(os)


class MemorylessPolicy(Policy):
    def action(self, o: Observation):
        raise NotImplementedError("action not implemented")

    def __getitem__(self, os: List[Observation]):
        return self.action(os[-1])

def make_memoryless(π: Policy):
    out = MemorylessPolicy()
    out.action = lambda os: π.action(os[-1])
    return out

# ------------------------------------------------------------------------------
# POMDP class with some logic implemented
# ------------------------------------------------------------------------------
class POMDP:
    """
    The POMDP class is a generic class that holds the logic for POMDPs. It's
    not meant to hold state that changes -- rather, the state should be recorded
    outside of the POMDP class and passed in as parameters to the methods.

    This class only holds the mathematical logic for the POMDP.
    """
    def __init__(self, discount: float):
        # discount factor
        self.discount = discount


    def _reward(self, s: State, a: Action, sp: State) -> float:
        """
        Returns the reward for taking action a in state s and transitioning to 
        state sp. This should be fully determined.
        """
        raise NotImplementedError("reward not implemented")


    def transition(self, s: State, a: Action) -> dict[State, float]:
        """
        Returns the probability of transitioning to state sp when taking action
        a in state s.
        """
        raise NotImplementedError("transition not implemented")
    

    def observation(self, a: Action, sp: State) -> Observation:
        """
        Returns the observation for taking action a and transitioning to state
        sp.
        """
        raise NotImplementedError("observation not implemented")
        

    def lookahead(self, s: State, a: Action, U: dict[State, float]) -> float:
        """
        Calculates the expected utility of taking action a from state s.

        params
            s -- the current state
            a -- action to take
            U -- a dictionary mapping states to their expected utility
        """
        return self.reward(s, a) + self.discount * sum(
            prob * U[sp] for (sp, prob) in self.transition(s, a).items()
        )


    def reward(self, s: State, a: Action) -> float:
        """
        Returns the expected reward for taking action a in state s.

        params:
            s -- the current state
            a -- the action performed from state s
        """
        out = 0
        for (sp, prob) in self.transition(s, a).items():
            out += prob * self._reward(s, a, sp)
        return out


    def rollout(self, s: State, o: Observation, π: Policy, d: int = 10) -> float:
        """
        Conducts a rollout from state s using policy π for d time steps.

        params
            s -- the initial state
            π -- the policy to use
            d -- the number of time steps to simulate
        """
        out = 0
        os = [o]

        for i in range(d):
            # decide which action to take
            a = π[os]

            # update the current reward
            out += (self.discount ** i) * self.reward(s, a)

            # calculate the next states we can transition into
            T = self.transition(s, a)
            if not T:
                break

            # randomly move in to one of the next states
            next_states = list(T.keys())
            next_state_ps = [T[sp] for sp in next_states]
            sp = np.random.choice(next_states, p=next_state_ps)

            # observe the next state
            os.append(self.observation(a, sp))
            s = sp
    
        return out