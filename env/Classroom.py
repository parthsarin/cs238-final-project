from .Student import Student, StudentState, StudentObservation, StudentAction
from .Teacher import Teacher, TeacherState, TeacherObservation, TeacherAction
from collections import defaultdict
from typing import List, Tuple, Dict, Set
import random


class Assignment:
    def __init__(self, difficulty: float, student_idx: float):
        assert 0 <= difficulty <= 1, "difficulty must be in [0, 1]"
        self.difficulty = difficulty
        self.student_idx = student_idx

        self.submitted = False
        self.time_submitted = None
        self.time_graded = None

    def submit(self, s: StudentState):
        """
        Calculates the quality of the assignment based on the student's g values
        and the time they worked on the assignment.
        """
        g = s.g
        time_worked = s.time_worked

        # also making quality = value between 0 - 100
        quality = sum(g) / len(g) + time_worked / 20
        quality *= 100
        self.quality = max(0, min(100, quality))

        self.submitted = True

    def grade(self, t: TeacherState):
        """
        Calculates the grade of the assignment based on the teacher's mh value
        and the assignment's difficulty.

        """
        assert self.submitted, "assignment must be submitted before grading"
        mh = t.mh  # between -1 and 1
        difficulty = self.difficulty  # between 0 and 1

        grade = self.quality
        grade += 15 * (mh + 1)  # so that mh > 0
        grade -= 15 * difficulty
        return max(0, min(100, grade))


class Classroom:
    """
    A simulation of the student-teacher interactions over time. This class is
    responsible for storing all of the assignments, the student and teacher
    states, and the student and teacher policies.
    """

    def __init__(self, n_students, assignment_every=3):
        # create objects to manage the student and teacher logic
        self.s_logic = Student()
        self.t_logic = Teacher()
        self.n_students = n_students

        # create the initial student states
        tmp = [self._initialize_student() for _ in range(n_students)]
        self.student_s = [s for (s, _) in tmp]
        self.student_o = [[o] for (_, o) in tmp]

        # ungraded, graded lists of assignments
        self.unsubmitted: Dict[int, Set[Assignment]] = defaultdict(set)
        self.ungraded: List[Assignment] = []
        self.graded: List[Assignment] = []
        self.assignment_every = assignment_every

        # create the initial teacher state
        s, o = self._initialize_teacher()
        self.teacher_s = s
        self.teacher_o = [o]

    def _initialize_student(self) -> Tuple[StudentState, StudentObservation]:
        """
        Creates an initial student.
        """
        initial_mh = random.uniform(-1, 1)
        initial_prod = random.uniform(0, 1)
        # adjust number of competencies via range(x)
        initial_g = [random.uniform(0, 1) for _ in range(8)]
        initial_free_time = round(random.uniform(0, 7))
        initial_time_worked = 0  # no time worked initially

        # create initial student state
        initial_student_state = StudentState(
            initial_mh, initial_prod, initial_g, initial_free_time, 0,
            initial_time_worked, []
        )

        # create initial student observation
        initial_observation = StudentObservation(None, initial_free_time, 0)

        return initial_student_state, initial_observation

    def _initialize_teacher(self) -> Tuple[TeacherState, TeacherObservation]:
        """
        Creates an initial teacher.
        """
        initial_mh = random.uniform(-1, 1)
        initial_prod = random.uniform(0, 1)
        initial_g = random.uniform(0, 1)
        initial_free_time = round(random.uniform(0, 7))
        initial_num_assignments = 0

        # create initial teacher state
        initial_teacher_state = TeacherState(initial_mh, initial_prod, initial_g, initial_free_time,
                                             initial_num_assignments)

        # create initial teacher observation
        initial_observation = TeacherObservation(
            initial_free_time, initial_num_assignments)

        return initial_teacher_state, initial_observation

    @staticmethod
    def _sample_from_weights_dict(weights_dict: dict[object, float]) -> object:
        """
        Samples an object from the weights_dict, where the weights_dict maps
        objects to their weights.
        """
        objects, weights = zip(*weights_dict.items())
        return random.choices(objects, weights)[0]

    def s_transition(self, s: StudentState, a: StudentAction) -> StudentState:
        """
        Returns the next state of the student after taking action a in state s.
        """
        return self._sample_from_weights_dict(self.s_logic.transition(s, a))

    def student_step(self, actions: List[StudentAction], t=None) -> List[float]:
        """
        Performs a step for the students. First calculates the next state using
        the student logic, then calculates the reward for each student, and
        finally calculates the observation for each student. The new states
        and observations are stored in the corresponding lists and the rewards
        are returned.

        If any of the students submit an assignment, the function will create
        a new assignment and append it to the corresponding student's list of
        assignments. This will also affect the teacher state and teacher
        observation.

        params:
            actions -- the actions that the students take, where actions[i] is
                       the action corresponding to student i

        returns:
            rewards -- the rewards for each student, where rewards[i] is the
                       reward corresponding to student i
        """
        if t is not None and t % self.assignment_every == 0:
            # create a new assignment
            difficulty = random.random()
            for s_idx in range(self.n_students):
                assign = Assignment(difficulty, s_idx)
                self.unsubmitted[s_idx].add(assign)

        rewards = []

        for s_idx, action in enumerate(actions):
            student_state = self.student_s[s_idx]

            # check if the student submitted an assignment
            if action.submit:
                # get the student's unsubmitted assignments
                unsubmitted = self.unsubmitted[s_idx]
                if not unsubmitted:
                    raise ValueError("student has no unsubmitted assignments")

                # submit the assignment
                assign = unsubmitted.pop()
                assign.submit(student_state)
                assign.time_submitted = t
                self.ungraded.append(assign)

            # new state
            new_student_state = self.s_transition(student_state, action)
            new_student_state.num_assignments = len(self.unsubmitted[s_idx])

            # calculate reward for the student
            reward = self.s_logic._reward(
                student_state, action, new_student_state)
            rewards.append(reward)

            # update student state
            self.student_s[s_idx] = new_student_state

            # sample an observation from the student's observation model
            self.student_o[s_idx].append(self._sample_from_weights_dict(
                self.s_logic.observation(action, new_student_state)
            ))

        return rewards

    def t_transition(self, s: TeacherState, a: TeacherAction) -> TeacherState:
        """
        Returns the next state of the teacher after taking action a in state s.
        """
        return self._sample_from_weights_dict(self.t_logic.transition(s, a))

    def teacher_step(self, a: TeacherAction, t=None) -> float:
        """
        Performs a step for the teacher. Updates self.teacher_s and appends the
        new observation to self.teacher_o. Also returns the reward that the
        teacher receives.

        params:
            a -- the action that the teacher takes

        returns:
            reward -- the reward that the teacher receives
        """
        # get most updated teacher state
        teacher_state = self.teacher_s

        # calculate next step and reward
        new_teacher_state = self.t_transition(teacher_state, a)
        new_teacher_state.num_assignments = len(self.ungraded)
        reward = self.t_logic._reward(teacher_state, a, new_teacher_state)

        # grade assignments
        num_assignments = self.t_logic._assignments_graded(teacher_state, a)
        for _ in range(num_assignments):
            try:
                # get the student and assignment
                assign = self.ungraded.pop(0)
            except IndexError:
                # no more assignments to grade
                break

            # grade the assignment
            grade = assign.grade(teacher_state)
            assign.time_graded = t
            self.graded.append(assign)

            # randomly select a competency to affect
            student_idx = assign.student_idx

            new_g = self.student_s[student_idx].g.copy()
            g_idx = random.randint(0, 7)
            new_g[g_idx] = min(1, new_g[g_idx] + (grade / 100) * 1e-4)

            self.student_s[student_idx].g = new_g

            # change the student observation so they can see their grade
            self.student_o[student_idx][-1].assignment_grade = grade

        # update teacher state and observation lists
        self.teacher_s = new_teacher_state
        self.teacher_o.append(self._sample_from_weights_dict(
            self.t_logic.observation(a, new_teacher_state)
        ))

        return reward
