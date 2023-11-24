from env import Classroom
import pandas as pd
from typing import List


class SimulationSnapshot:
    """
    A snapshot of the classroom at a given time.
    """

    def __init__(self, c: Classroom, t: int, **kwargs):
        self.t = t

        # count of values
        self.n_ungraded = len(c.ungraded)
        self.n_graded = len(c.graded)
        self.n_students = c.n_students

        # grading gap
        self.grading_gap = None
        gg_tmp = [a.time_graded - a.time_submitted for a in c.graded]
        if gg_tmp:
            self.grading_gap = sum(gg_tmp) / len(gg_tmp)

        # teacher state
        self.teacher_s = c.teacher_s.__dict__
        self.teacher_o = c.teacher_o[-1].__dict__
        self.teacher_a = kwargs['teacher_a'].__dict__ if 'teacher_a' in kwargs else None
        self.teacher_r = kwargs['teacher_r'] if 'teacher_r' in kwargs else None

        # student states
        self.student_s = [s.__dict__ for s in c.student_s]
        self.student_o = [o[-1].__dict__ for o in c.student_o]
        self.student_a = [
            a.__dict__ for a in kwargs['student_as']
        ] if 'student_as' in kwargs else None
        self.student_r = kwargs['student_rs'] if 'student_rs' in kwargs else None

        # average rewards
        self.avg_sr = None
        if self.student_r:
            self.avg_sr = sum(self.student_r) / len(self.student_r)

    def display(self):
        print(
            f"[day {self.t}] avg student reward: {self.avg_sr:.2f}, teacher reward: {self.teacher_r:.2f}")
        print(f"\tnum graded: {self.n_graded}")
        print(f"\tnum ungraded: {self.n_ungraded}")
        if self.grading_gap:
            print(f"\tavg grading gap: {self.grading_gap:.2f}")


class Log:
    """
    A class that logs the results of a simulation over time and exports useful
    statistics about the simulation.
    """

    def __init__(self, classroom: Classroom):
        self.c = classroom
        self.history: List[SimulationSnapshot] = []

    def record(self, t: int, **kwargs):
        """
        Logs the current state of the classroom.
        """
        self.history.append(SimulationSnapshot(self.c, t, **kwargs))

    def display_latest(self):
        """
        Displays the latest snapshot of the classroom.
        """
        self.history[-1].display()

    @staticmethod
    def _prefix_dict(d: dict, prefix: str):
        return {f"{prefix}_{k}": v for k, v in d.items()}

    def t_oar_memoryless(self):
        """
        Returns a list of tuples (o, a, r) for the teacher at each time step,
        observation o, action a, and reward r. This function only returns the
        latest observation at time t.
        """
        data = [
            {
                **self._prefix_dict(self.history[i-1].teacher_o, 'o'),
                **self._prefix_dict(self.history[i].teacher_a, 'a'),
                'r': self.history[i].teacher_r,
                **self._prefix_dict(self.history[i].teacher_o, 'op'),
                'day': i,
            }
            for i in range(1, len(self.history))
        ]
        return pd.DataFrame(data)

    def s_oar_memoryless(self):
        """
        Returns a list of tuples (o, a, r) for each student at each time step,
        observation o, action a, and reward r. This function only returns the
        latest observation at time t.
        """
        return pd.DataFrame([
            {
                **self._prefix_dict(tup[0], 'o'),
                **self._prefix_dict(tup[1], 'a'),
                'r': tup[2],
                **self._prefix_dict(tup[3], 'op'),
                'day': i,
            }
            for i in range(1, len(self.history))
            for tup in zip(
                self.history[i-1].student_o,
                self.history[i].student_a,
                self.history[i].student_r,
                self.history[i].student_o
            )
        ])

    def t_oar(self):
        """
        Returns a list of tuples (o, a, r) for the teacher at each time step,
        observation o, action a, and reward r.
        """
        obs = [s.teacher_o for s in self.history]
        return [
            {
                'os': obs[:i],
                **self._prefix_dict(s.teacher_a, 'a'),
                'r': s.teacher_r,
                'op': obs[i],
                'day': s.t,
            }
            for i, s in enumerate(self.history[1:])
        ]

    def s_oar(self):
        """
        Returns a list of tuples (o, a, r) for each student at each time step,
        observation o, action a, and reward r.
        """
        obs = [s.student_o for s in self.history]
        return [
            {
                'os': tup[0],
                **self._prefix_dict(tup[1], 'a'),
                'r': tup[2],
                'op': tup[3],
                'day': i,
            }
            for i, s in enumerate(self.history[1:])
            for tup in zip(
                obs[:i],
                s.student_a,
                s.student_r,
                obs[i]
            )
        ]
