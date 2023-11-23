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

        # add code here
        self.submitted = True
        quality = 1 - (time_worked - sum(g))/time_worked
        quality *= 100
        return max(0, min(100, quality)) # also making quality = value between 0 - 100
    
    def grade(self, t: TeacherState):
        """
        Calculates the grade of the assignment based on the teacher's mh value
        and the assignment's difficulty.

        """
        assert self.submitted, "assignment must be submitted before grading"
        mh = t.mh # between -1 and 1
        difficulty = self.difficulty # between 0 and 1

        grade = 70
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

    
    def _initialize_student(self) -> Tuple[StudentState, StudentObservation]:
        """
        Creates an initial student.
        """
        initial_mh = random.uniform(-1, 1)
        initial_prod = random.uniform(0, 1)
        initial_g = [random.uniform(0, 1) for _ in range(8)]  # adjust number of competencies via range(x)
        initial_free_time = random.uniform(0, 7)
        initial_time_worked = 0  # no time worked initially

        # create initial student state
        initial_student_state = StudentState(initial_mh, initial_prod, initial_g, initial_free_time,
                                             initial_time_worked)

        # create initial student observation
        initial_observation = StudentObservation(0, initial_free_time)  # initial grade set to 0

        return initial_student_state, initial_observation

    def _initialize_teacher(self) -> Tuple[TeacherState, TeacherObservation]:
        """
        Creates an initial teacher.
        """
        initial_mh = random.uniform(-1, 1)
        initial_prod = random.uniform(0, 1)
        initial_g = random.uniform(0, 1)
        initial_free_time = random.uniform(0, 7)
        initial_num_assignments = self.calculate_number_of_assignments()

        # create initial teacher state
        initial_teacher_state = TeacherState(initial_mh, initial_prod, initial_g, initial_free_time,
                                             initial_num_assignments)

        # create initial teacher observation
        initial_observation = TeacherObservation(initial_free_time, initial_num_assignments)

        return initial_teacher_state, initial_observation


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
            student_observation = self.student_o[i][-1]  # Get the latest observation

            # student step
            new_student_state_dict = self.s_logic.step(student_state, action)
            new_student_state, _ = next(iter(new_student_state_dict.items()))

            # check if the student submitted an assignment
            if action.submit:
                # create new assignment and add to student's list of assignments
                assignment_quality = self.s_logic.submit(new_student_state)
                # initialize difficulty as random value between 0 and 1
                difficulty = random()
                new_assignment = Assignment(assignment_quality, difficulty)
                self.student_s.assignments[i].append(new_assignment)

                # update teacher state and observation
                self.teacher_s.num_assignments -= 1
                self.teacher_o[0].num_assignments = self.teacher_s.num_assignments

            # calculate reward for the student
            reward = self.s_logic.calculate_reward(student_state, action, new_student_state)
            rewards.append(reward)

            # update student state and observation lists
            self.student_s[i] = new_student_state
            self.student_o[i].append(self.s_logic.observe(action, new_student_state))

        return rewards
    
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
        new_teacher_state_dict = self.t_logic.step(teacher_state, a)
        new_teacher_state, _ = next(iter(new_teacher_state_dict.items()))

        # calculate reward for teacher
        reward = self.t_logic.calculate_reward(teacher_state, a, new_teacher_state)

        # update teacher state and observation lists
        self.teacher_s = new_teacher_state
        self.teacher_o.append(self.t_logic.observe(a, new_teacher_state))

        return reward