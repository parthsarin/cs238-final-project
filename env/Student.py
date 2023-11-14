from .POMDP import POMDP, State, Action, Observation
from typing import Union, List

# ------------------------------------------------------------------------------
# state, action, and observation spaces
# ------------------------------------------------------------------------------
class StudentAction(Action):
    """
    Defines the action space for the student. Students can either submit or not
    submit. If they choose to submit, that's it. If they choose not to submit,
    they must choose how to allocate their remaining time between resting and
    working (numbers which add up to 1).
    """
    def __init__(
            self, 
            submit: bool = False,
            rest: Union[float, None] = None, 
            work: Union[float, None] = None
        ):
        super().__init__()
        self.submit = submit

        if not submit:
            assert 0 <= rest <= 1, "rest must be in [0, 1]"
            assert 0 <= work <= 1, "work must be in [0, 1]"
            assert rest + work == 1, "rest + work must be = 1"

        self.rest = rest
        self.work = work

    def __hash__(self):
        return hash((self.submit, self.rest, self.work))
    
    def __eq__(self, other: object) -> bool:
        if self.submit and other.submit:
            return True
        
        return self.rest == other.rest and self.work == other.work

class StudentState(State):
    """
    Defines the state space for the student. The state space is defined by the
    student's current mental health (mh), productivity (prod), and a measure of
    their competencies/skills (g). Of course, this is an inaccurate way to
    represent intelligence, meant as a critique of the way we measure it in
    schools.

    mh -- mental health, in [-1, 1]
    prod -- productivity, in [0, 1]
    g -- list of competencies, in [0, 1]
    free_time -- free time, in [0, 7]
    time_worked -- the total amount of time spent working on the current assignment
    """
    def __init__(self, mh, prod, g: List[float], free_time, time_worked):
        super().__init__()

        assert -1 <= mh <= 1, "mh must be in [-1, 1]"
        assert 0 <= prod <= 1, "prod must be in [0, 1]"
        assert all([0 <= gi <= 1 for gi in g]), "g must be in [0, 1]"
        assert 0 <= free_time <= 7, "free_time must be in [0, 7]"

        self.mh = mh
        self.prod = prod
        self.g = g
        self.free_time = free_time
        self.time_worked = time_worked
    
    def __hash__(self):
        return hash((self.mh, self.prod, tuple(self.g), self.free_time, self.time_worked))
    
    def __eq__(self, other: object) -> bool:
        return all([
            self.mh == other.mh, 
            self.prod == other.prod, 
            self.g == other.g, 
            self.free_time == other.free_time,
            self.time_worked == other.time_worked
        ])


class StudentObservation(Observation):
    """
    The information that the student observes from the environment at every
    time step.
    """
    def __init__(self, assignment_grade, free_time):
        super().__init__()

        assert 0 <= assignment_grade <= 100, "assignment_grade must be in [0, 100]"
        assert 0 <= free_time <= 7, "free_time must be in [0, 7]"

        self.assignment_grade = assignment_grade
        self.free_time = free_time
    
    def __hash__(self):
        return hash((self.assignment_grade, self.free_time))
    
    def __eq__(self, other: object) -> bool:
        return self.assignment_grade == other.assignment_grade and self.free_time == other.free_time


# ------------------------------------------------------------------------------
# student class and logic
# ------------------------------------------------------------------------------
class Student(POMDP):
    def __init__(self):
        super().__init__(0.95)

    
    def _reward(self, s: StudentState, a: StudentAction, sp: StudentState) -> float:
        """
        Returns the reward for taking action a in state s and transitioning to 
        state sp. This should be fully determined.
        """
        raise NotImplementedError("reward not implemented")
    

    def transition(self, s: StudentState, a: StudentAction) -> dict[StudentState, float]:
        """
        Returns the probability of transitioning to state sp when taking action
        a in state s.
        """
        raise NotImplementedError("transition not implemented")

    
    def observation(self, a: StudentAction, sp: StudentState) -> Observation:
        """
        Returns the observation for taking action a and transitioning to state
        sp.
        """
        raise NotImplementedError("observation not implemented")