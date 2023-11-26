from random import random
import numpy as np

from env import (
    MemorylessPolicy,
    TeacherObservation,
    StudentObservation,
    StudentAction,
    TeacherAction,
    Observation,
    Action
)
from numeric import full_round


class StudentPolicy(MemorylessPolicy):
    def __init__(self, submit_thresh: float = 0.3):
        self.submit_thresh = submit_thresh

    def action(self, o: StudentObservation):
        if o.num_assignments > 0 and random() < self.submit_thresh:
            return StudentAction(submit=True)

        rest = random()
        work = 1 - rest
        rest, work = full_round((rest, work), 1)
        return StudentAction(rest=rest, work=work)


class TeacherPolicy(MemorylessPolicy):
    def action(self, o: TeacherObservation):
        a = np.random.dirichlet((1, 1, 1))
        a = full_round(a, 1)
        return TeacherAction(*a)


def rand_action(o: Observation, act_class: type(Action)):
    """
    Returns a random action for the given observation.
    """
    if act_class is StudentAction:
        if o.num_assignments > 0 and random() < 0.3:
            return StudentAction(submit=True)

        rest = random()
        work = 1 - rest
        rest, work = full_round((rest, work), 1)
        return StudentAction(rest=rest, work=work)

    elif act_class is TeacherAction:
        a = np.random.dirichlet((1, 1, 1))
        a = full_round(a, 1)
        return TeacherAction(*a)
