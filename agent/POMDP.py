from collections import defaultdict

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

class POMDP:
    def __init__(self):
        # state space, action space
        self.S = None
        self.A = None
        pass
    
    def lookahead(self, b: defaultdict, a: Action) -> float:
        """
        Calculates the expected utility of taking action a in belief state b.

        params
            b -- belief state (supports b[s] to lookup probability of state s)
            a -- action to take

        returns
        expected utility of taking action a in belief state b
        """
        raise NotImplementedError("lookahead not implemented")
    
    def reward(self, s: State, a: Action, sp: State) -> float:
        """
        Returns the reward for taking action a in state s and transitioning to 
        state sp. This should be fully determined.
        """
        raise NotImplementedError("reward not implemented")
    
    def transition(self, s: State, a: Action, sp: State) -> dict[State, float]:
        """
        Returns the probability of transitioning to state sp when taking action
        a in state s.
        """
        raise NotImplementedError("transition not implemented")
