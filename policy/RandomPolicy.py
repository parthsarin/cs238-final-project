from random import random
import numpy as np

from env import MemorylessPolicy, Observation, StudentAction, TeacherAction

class StudentPolicy(MemorylessPolicy):
    def __init__(self, submit_thresh: float = 0.3):
        self.submit_thresh = submit_thresh

    def action(self, o: Observation):
        if random() < self.submit_thresh:
            return StudentAction(submit = True)
        
        rest = random()
        work = 1 - rest
        return StudentAction(rest = rest, work = work)


class TeacherPolicy(MemorylessPolicy):
    def action(self, o: Observation):
        a = np.random.dirichlet((1, 1, 1))
        return TeacherAction(*a)