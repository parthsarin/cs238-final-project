from .Student import Student, StudentState, StudentObservation, StudentAction
from .Teacher import Teacher, TeacherState, TeacherObservation, TeacherAction
from typing import List, Tuple

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
        raise NotImplementedError("submit not implemented")
    
    def grade(self, t: TeacherState):
        """
        Calculates the grade of the assignment based on the teacher's mh value
        and the assignment's difficulty.
        """
        assert self.submitted, "assignment must be submitted before grading"
        raise NotImplementedError("grade not implemented")


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
        raise NotImplementedError("initialize_student not implemented")
    

    def _initialize_teacher(self) -> Tuple[TeacherState, TeacherObservation]:
        """
        Creates an initial teacher.
        """
        raise NotImplementedError("initialize_teacher not implemented")


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
        raise NotImplementedError("student_step not implemented")
    
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
        raise NotImplementedError("teacher_step not implemented")