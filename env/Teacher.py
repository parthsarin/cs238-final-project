from .POMDP import POMDP, State, Action, Observation

# ------------------------------------------------------------------------------
# state, action, and observation space
# ------------------------------------------------------------------------------
class TeacherAction(Action):
    """
    Teachers allocate a certain amount of time to resting, grading, and
    professional development (pd). These numbers must add up to 1.
    """
    def __init__(self, rest, grading, pd):
        assert all([0 <= x <= 1 for x in [rest, grading, pd]]), "time allocations must be in [0, 1]"
        assert rest + grading + pd == 1, "time allocations must add up to 1"

        self.rest = rest
        self.grading = grading
        self.pd = pd

    def __hash__(self):
        return hash((self.rest, self.grading, self.pd))

    def __eq__(self, other: object) -> bool:
        return self.rest == other.rest and self.grading == other.grading and self.pd == other.pd
    

class TeacherState(State):
    """
    Teachers have a certain amount of mental health (mh), productivity (prod),
    competence (g), and free time (free_time). These are all floats in the
    following ranges:
        mh -- [-1, 1]
        prod -- [0, 1]
        g -- [0, 1]
        free_time -- [0, 7]
    
    The free time is fully observable, but the other three are not.
    """
    def __init__(self, mh, prod, g: float, free_time):
        assert -1 <= mh <= 1, "mh must be in [-1, 1]"
        assert 0 <= prod <= 1, "prod must be in [0, 1]"
        assert 0 <= g <= 1, "g must be in [0, 1]"
        assert 0 <= free_time <= 7, "free_time must be in [0, 7]"

        self.mh = mh
        self.prod = prod
        self.g = g
        self.free_time = free_time
    
    def __hash__(self):
        return hash((self.mh, self.prod, self.g, self.free_time))

    def __eq__(self, other: object) -> bool:
        return all([
            self.mh == other.mh,
            self.prod == other.prod,
            self.g == other.g,
            self.free_time == other.free_time
        ])
    
class TeacherObservation(Observation):
    """
    Teachers can observe the amount of free time they have and the student
    assignments they have to grade.
    """
    def __init__(self, free_time, num_assignments):
        assert 0 <= free_time <= 7, "free_time must be in [0, 7]"
        assert 0 <= num_assignments <= 500, "num_assignments must be in [0, 500]"

        self.free_time = free_time
        self.num_assignments = num_assignments
    
    def __hash__(self):
        return hash((self.free_time, self.num_assignments))

    def __eq__(self, other: object) -> bool:
        return all([
            self.free_time == other.free_time,
            self.num_assignments == other.num_assignments
        ])

# ------------------------------------------------------------------------------
# teacher class and logic
# ------------------------------------------------------------------------------
class Teacher(POMDP):
    def __init__(self):
        super().__init__(0.95)
    
    def _reward(self, s: TeacherState, a: TeacherAction, sp: TeacherState) -> float:
        """
        The reward that the teacher gets from taking action a in state s and
        transitioning to state sp.
        """
        raise NotImplementedError("reward not implemented")

    def transition(self, s: TeacherState, a: TeacherAction) -> dict[TeacherState, float]:
        """
        The next state transition probabilities for taking action a in state s.
        """
        raise NotImplementedError("transition not implemented")
    
    def observation(self, a: TeacherAction, sp: TeacherState) -> Observation:
        """
        Returns the observation for taking action a and transitioning to state
        sp.
        """
        raise NotImplementedError("observation not implemented")