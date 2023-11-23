from .Student import Student, StudentState, StudentObservation, StudentAction
from .Teacher import Teacher, TeacherState, TeacherObservation, TeacherAction
from typing import List, Tuple
import numpy as np
import random

class Assignment:
    def __init__(self, difficulty: float):
        assert 0 <= difficulty <= 1, "difficulty must be in [0, 1]"
        self.difficulty = difficulty
        self.submitted = False
    
    def submit(self, s: StudentState):
        """
        Calculates the quality of the assignment based on the student's g values
        and the time they worked on the assignment.
        """
        g = s.g
        time_worked = s.time_worked

        # also making quality = value between 0 - 100
        quality = 1 - (time_worked - sum(g))/time_worked
        quality *= 100
        self.quality = max(0, min(100, quality))

        self.submitted = True
    
    def grade(self, t: TeacherState):
        """
        Calculates the grade of the assignment based on the teacher's mh value
        and the assignment's difficulty.

        """
        assert self.submitted, "assignment must be submitted before grading"
        mh = t.mh # between -1 and 1
        difficulty = self.difficulty # between 0 and 1

        grade = self.quality
        grade += 15 * (mh + 1) # so that mh > 0
        grade -= 15 * difficulty
        return max(0, min(100, grade))

class Classroom:
    """
    A simulation of the student-teacher interactions over time. This class is
    responsible for storing all of the assignments, the student and teacher
    states, and the student and teacher policies.
    """
    def __init__(self, n_students):
        # create objects to manage the student and teacher logic
        self.s_logic = Student()
        self.t_logic = Teacher()
        self.n_students = n_students

        # create the initial student states
        tmp = [self._initialize_student() for _ in range(n_students)]
        self.student_s = [s for (s, _) in tmp]
        self.student_o = [[o] for (_, o) in tmp]
        self.assignments = [[] for _ in range(n_students)]

        # create the initial teacher state
        s, o = self._initialize_teacher()
        self.teacher_s = s
        self.teacher_o = [o]
        self.ungraded_assignments = 0

    
    def _initialize_student(self) -> Tuple[StudentState, StudentObservation]:
        """
        Creates an initial student.
        """
        initial_mh = random.uniform(-1, 1)
        initial_prod = random.uniform(0, 1)
        initial_g = [random.uniform(0, 1) for _ in range(8)]  # adjust number of competencies via range(x)
        initial_free_time = round(random.uniform(0, 7))
        initial_time_worked = 0  # no time worked initially

        # create initial student state
        initial_student_state = StudentState(
            initial_mh, initial_prod, initial_g, initial_free_time,
            initial_time_worked, []
        )

        # create initial student observation
        initial_observation = StudentObservation(None, initial_free_time)

        return initial_student_state, initial_observation

    def _initialize_teacher(self) -> Tuple[TeacherState, TeacherObservation]:
        """
        Creates an initial teacher.
        """
        initial_mh = random.uniform(-1, 1)
        initial_prod = random.uniform(0, 1)
        initial_g = random.uniform(0, 1)
        initial_free_time = random.uniform(0, 7)
        initial_num_assignments = 0

        # create initial teacher state
        initial_teacher_state = TeacherState(initial_mh, initial_prod, initial_g, initial_free_time,
                                             initial_num_assignments)

        # create initial teacher observation
        initial_observation = TeacherObservation(initial_free_time, initial_num_assignments)

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
        new_mh = max(-1, min(1, s.mh + (a.rest - a.work) * 0.1))
        new_prod = max(0, min(1, s.prod + a.work * 0.1))
        new_g = [min(1, gi + a.work * 0.05) for gi in s.g]
        new_assign_durations = s.assign_durations

        if a.submit:
            new_time_worked = 0
            new_assign_durations.append(s.time_worked)
        else:
            new_time_worked = s.time_worked + a.work * s.free_time

        return StudentState(
            new_mh, new_prod, new_g, random.randint(0, 7), 
            new_time_worked, new_assign_durations
        )


    def student_step(self, actions: List[StudentAction]) -> List[float]:
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
        rewards = []

        for i, action in enumerate(actions):
            student_state = self.student_s[i]

            # student step
            new_student_state = self.s_transition(student_state, action)

            # check if the student submitted an assignment
            if action.submit:
                # create new assignment and add to student's list of assignments
                assign = Assignment(random())
                assign.submit(student_state)

                # log the assignments
                self.assignments[i].append(assign)
                self.ungraded_assignments += 1

            # calculate reward for the student
            reward = self.s_logic._reward(student_state, action, new_student_state)
            rewards.append(reward)

            # update student state
            self.student_s[i] = new_student_state

            # sample an observation from the student's observation model
            self.student_o[i].append(self._sample_from_weights_dict(
                self.s_logic.observation(action, new_student_state)
            ))

        return rewards
    
    def t_transition(self, s: TeacherState, a: TeacherAction) -> TeacherState:
        """
        Returns the next state of the teacher after taking action a in state s.
        """
        new_mh = max(-1, min(1, s.mh + a.rest * 0.1 - a.grading * 0.05))
        new_prod = max(0, min(1, s.prod + a.grading * 0.1))
        new_g = max(0, min(1, s.g + a.pd * 0.1))

        time_grading = a.grading * s.free_time
        assignments_per_hour = 1 + s.g * 2
        assignments_graded = round(time_grading * assignments_per_hour)
        new_num_assignments = max(0, s.num_assignments - assignments_graded)

        return TeacherState(
            new_mh, new_prod, new_g, random.randint(0, 7), 
            new_num_assignments
        )

    def teacher_step(self, a: TeacherAction) -> float:
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

        # teacher step
        new_teacher_state = self.t_transition(teacher_state, a)

        # calculate reward for teacher
        reward = self.t_logic._reward(teacher_state, a, new_teacher_state)

        # update teacher state and observation lists
        self.teacher_s = new_teacher_state
        self.teacher_o.append(self._sample_from_weights_dict(
            self.t_logic.observation(a, new_teacher_state)
        ))

        return reward