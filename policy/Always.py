from random import random
import numpy as np

from env import (
    MemorylessPolicy,
    TeacherObservation,
    StudentObservation,
    StudentAction,
    TeacherAction
)


class StudentAlwaysWork(MemorylessPolicy):
    def __init__(self, submit_thresh: float = 0.3):
        self.submit_thresh = submit_thresh

    def action(self, o: StudentObservation):
        if o.num_assignments > 0 and random() < self.submit_thresh:
            return StudentAction(submit=True)

        return StudentAction(rest=0, work=1)


class StudentAlwaysRest(MemorylessPolicy):
    def __init__(self, submit_thresh: float = 0.3):
        self.submit_thresh = submit_thresh

    def action(self, o: StudentObservation):
        return StudentAction(rest=1, work=0)


class TeacherAlwaysGrade(MemorylessPolicy):
    def action(self, o: TeacherObservation):
        return TeacherAction(0, 1, 0)


class TeacherAlwaysRest(MemorylessPolicy):
    def action(self, o: TeacherObservation):
        return TeacherAction(1, 0, 0)
