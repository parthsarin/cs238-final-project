from .POMDP import POMDP, State, Action, Observation
from typing import Union, List
import numpy as np
from itertools import product


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
    assign_durations -- a list of the total amount of time spent working on each assignment
    """

    def __init__(
        self, mh: float, prod: float, g: List[float],
        free_time: int,
        num_assignments: int,
        time_worked: float,
        assign_durations: List[float]
    ):
        super().__init__()

        assert -1 <= mh <= 1, "mh must be in [-1, 1]"
        assert 0 <= prod <= 1, "prod must be in [0, 1]"
        assert all([0 <= gi <= 1 for gi in g]), "g must be in [0, 1]"
        assert 0 <= free_time <= 7, "free_time must be in [0, 7]"

        self.mh = mh
        self.prod = prod
        self.g = g
        self.free_time = free_time
        self.num_assignments = num_assignments
        self.time_worked = time_worked
        self.assign_durations = assign_durations

    def __hash__(self):
        return hash((
            self.mh, self.prod, tuple(self.g), self.free_time,
            self.num_assignments,
            self.time_worked, tuple(self.assign_durations)
        ))

    def __eq__(self, other: object) -> bool:
        return all([
            self.mh == other.mh,
            self.prod == other.prod,
            self.g == other.g,
            self.free_time == other.free_time,
            self.num_assignments == other.num_assignments,
            self.time_worked == other.time_worked,
            self.assign_durations == other.assign_durations
        ])

    def __repr__(self):
        # round the values to 3 decimal places
        return f"StudentState(mh = {round(self.mh, 3)}, prod = {round(self.prod, 3)}, g = {list(map(lambda x: round(x, 3), self.g))}, ft = {self.free_time}, n_assign = {self.num_assignments}, tw = {round(self.time_worked, 3)}, ad = {list(map(lambda x: round(x, 3), self.assign_durations))})"


class StudentObservation(Observation):
    """
    The information that the student observes from the environment at every
    time step.
    """

    def __init__(self, assignment_grade, free_time, num_assignments):
        super().__init__()

        assert assignment_grade is None or 0 <= assignment_grade <= 100, "assignment_grade must be None or in [0, 100]"
        assert 0 <= free_time <= 7, "free_time must be in [0, 7]"

        self.assignment_grade = assignment_grade
        self.free_time = free_time
        self.num_assignments = num_assignments

    def __hash__(self):
        return hash((self.assignment_grade, self.free_time, self.num_assignments))

    def __eq__(self, other: object) -> bool:
        return all((
            self.assignment_grade == other.assignment_grade,
            self.free_time == other.free_time,
            self.num_assignments == other.num_assignments
        ))

    def __repr__(self):
        return f"StudentObservation(grade = {self.assignment_grade}, ft = {self.free_time}, n_assign = {self.num_assignments})"


# ------------------------------------------------------------------------------
# student class and logic
# ------------------------------------------------------------------------------
class Student(POMDP):
    MH_PROD_COVARIANCE = np.array([
        [0.1, -0.05],
        [-0.05, 0.1]
    ])

    def __init__(self):
        super().__init__(0.95)

    def _reward(self, s: StudentState, a: StudentAction, sp: StudentState) -> float:
        """
        Returns the reward for taking action a in state s and transitioning to 
        state sp. Relative to |S|, |A|, and |O|, this should be O(1).
        """
        # Reward based on mental health improvement
        mh_improvement = sp.mh - s.mh
        # Reward for maintaining or improving productivity
        prod_reward = sp.prod - s.prod
        # Reward based on competencies improvement
        competencies_improvement = sum(sp.g) - sum(s.g)

        # penalty for too many assignments
        overwhelmed_penalty = 0
        if sp.num_assignments > 5:
            overwhelmed_penalty = (sp.num_assignments - 5) * -0.1

        # Aggregate the rewards
        return mh_improvement + prod_reward + competencies_improvement + overwhelmed_penalty

    def transition(self, s: StudentState, a: StudentAction) -> dict[StudentState, float]:
        """
        Returns the probability of transitioning to state sp when taking action
        a in state s. Relative to |S|, |A|, and |O|, this should be O(1).
        """
        if a.submit and s.num_assignments <= 0:
            raise ValueError("Cannot submit when there are no assignments")

        if a.submit:
            return {
                StudentState(
                    max(-1, min(1, s.mh + 0.01)),
                    s.prod,
                    s.g,
                    ft,
                    s.num_assignments - 1,
                    0,
                    s.assign_durations + [s.time_worked]
                ): 1/8
                for ft in range(8)
            }

        # rest improves mh, working improves productivity
        new_mh = max(-1, min(1, s.mh + a.rest * 0.1))
        new_prod = max(0, min(1, s.prod + a.work * 0.1))

        # take 10 samples from a multivariate normal distribution
        mh_prod = np.random.multivariate_normal(
            [new_mh, new_prod], self.MH_PROD_COVARIANCE, 10
        )
        np.clip(mh_prod[:, 0], -1, 1, out=mh_prod[:, 0])
        np.clip(mh_prod[:, 1], 0, 1, out=mh_prod[:, 1])

        # working improves competencies
        new_g = [min(1, gi + a.work * 0.05) for gi in s.g]

        # add on the time worked
        new_assign_durations = s.assign_durations.copy()
        new_time_worked = s.time_worked + a.work * s.free_time

        # 8 possible free time values, each with equal probability
        return {
            StudentState(
                mhp_sample[0], mhp_sample[1], new_g, ft, s.num_assignments,
                new_time_worked, new_assign_durations
            ): 1/8
            for mhp_sample, ft in product(mh_prod, range(8))
        }

    def observation(self, a: StudentAction, sp: StudentState) -> dict[StudentObservation, float]:
        """
        Returns the probability of each observation for taking action a and 
        transitioning to state sp. Relative to |S|, |A|, and |O|, this should be
        O(1).
        """
        if a.submit:
            # Assume the grade is a function of competencies and time worked
            time_worked = sp.assign_durations[-1]
            quality = sum(sp.g) + time_worked
            quality *= 100
            quality = max(0, min(100, quality))

            # the mean grade
            grade = max(0, min(100, quality))

            # draw 20 samples from a normal distribution with noise
            noisy_grades = np.random.normal(grade, 10, 20)

            # clip the grades to be in [0, 100]
            noisy_grades = np.clip(noisy_grades, 0, 100)

            return {
                StudentObservation(grade, sp.free_time, sp.num_assignments): 1/20
                for grade in noisy_grades
            }

        # otherwise, it's just the free time
        return {StudentObservation(None, sp.free_time, sp.num_assignments): 1.0}
