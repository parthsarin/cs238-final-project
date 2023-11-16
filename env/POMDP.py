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


class StateUtilityFunction:
    def __getitem__(self, s: State):
        raise NotImplementedError("utility not implemented")
    
    def __setitem__(self, s: State, u: float):
        raise NotImplementedError("utility not implemented")

class UtilityFunction:
    def __getitem__(self, b):
        raise NotImplementedError("utility not implemented")
    
    def __setitem__(self, b, u: float):
        raise NotImplementedError("utility not implemented")

def make_belief_utility(U: StateUtilityFunction):
    def get(self, b):
        states = list(b.keys())
        probs = np.array([b[s] for s in states])
        utils = np.array([U[s] for s in states])
        return np.dot(probs, utils)
    
    def set(self, b, u: float):
        states = list(b.keys())
        probs = np.array([b[s] for s in states])
        utils = np.array([U[s] for s in states])
        for i in range(len(states)):
            U[states[i]] = u * probs[i] / utils[i]

    out = UtilityFunction()
    out.__getitem__ = lambda b: get(out, b)
    out.__setitem__ = lambda b, u: set(out, b, u)
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
    

    def observation(self, a: Action, sp: State) -> dict[Observation, float]:
        """
        Returns the probability of each observation for taking action a and 
        transitioning to state sp.
        """
        raise NotImplementedError("observation not implemented")
        

    def lookahead_state(self, s: State, a: Action, U: StateUtilityFunction) -> float:
        """
        Calculates the expected utility of taking action a from state s.

        params
            s -- the current state
            a -- action to take
            U -- a StateUtilityFunction mapping states to their expected utility
        """
        return self.reward(s, a) + self.discount * sum(
            prob * U[sp] for (sp, prob) in self.transition(s, a).items()
        )
    

    def lookahead(self, b, a: Action, U: UtilityFunction, update):
        """
        Looks ahead using the transition and observation model and calculates
        the expected utility of taking action a from belief state b.

        params:
            b -- the current belief state
            a -- action to take
            U -- a UtilityFunction mapping beliefs to their expected utility
            update -- a function that accepts b, a, o and returns the new belief
                      state
        """
        def obs_prob(o, b, a):
            """
            The probability of observation o given belief state b and action a.
            """
            # we could actually be in a number of states
            states = list(b.keys())
            beliefs = np.array([b[s] for s in states])

            # the probability of observing o from each of the states...
            probs = []
            for s in states:
                # ...depends on the transition model...
                T = self.transition(s, a)
                sps = list(T.keys())
                t_probs = np.array([T[sp] for sp in sps])

                # ...and the observation model.
                o_probs = np.array([self.observation(a, sp)[o] for sp in sps])
                probs.append(np.dot(t_probs, o_probs))

            return np.dot(beliefs, probs)

        # calculate the support of the observation distribution
        O = set()
        for s in b:
            O |= set(self.observation(a, s).keys())

        # calculate the expected utility
        return self.reward(b, a) + self.discount * sum(
            obs_prob(o, b, a) * U[update(b, a, o)] for o in O
        )


    def reward_state(self, s: State, a: Action) -> float:
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


    def reward(self, b, a: Action) -> float:
        """
        Returns the expected reward given the belief state b and action a.

        params:
            b -- our beliefs about the current state
            a -- the action performed in this belief state
        """
        states = b.keys()
        probs = np.array([b[s] for s in states])
        rewards = np.array([self.reward_state(s, a) for s in states])
        return np.dot(probs, rewards)


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