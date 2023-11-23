from sys import argv
from env import Classroom, Policy
from evaluate import Log

def start_simulation(
        n_students: float,
        d: int,
        sπ: Policy,
        tπ: Policy
    ):
    """
    Starts the simulation with the given number of students and the given
    policies for the student and teacher.

    params:
        n_students -- the number of students in the classroom
        d -- the number of time steps / days to simulate
        sπ -- the student policy
        tπ -- the teacher policy
    """
    c = Classroom(n_students)
    l = Log(c)
    l.log(-1) # log initial state

    for t in range(d):
        # 1. student actions
        student_as = [sπ[o] for o in c.student_o]
        student_rs = c.student_step(student_as, t)
        
        # 2. teacher actions
        teacher_a = tπ[c.teacher_o]
        teacher_r = c.teacher_step(teacher_a, t)

        # 3. log results
        l.log(
            t, 
            teacher_a = teacher_a, 
            teacher_r = teacher_r, 
            student_as = student_as,
            student_rs = student_rs
        )
    print(l.s_oar_memoryless())



def main(strategy):
    strategy = strategy.lower()
    if strategy == 'random':
        from policy.RandomPolicy import StudentPolicy, TeacherPolicy
    else:
        raise ValueError(f"Unrecognized strategy: {strategy}")
    
    sπ = StudentPolicy()
    tπ = TeacherPolicy()

    start_simulation(35, 365, sπ, tπ)


if __name__ == '__main__':
    if len(argv) != 2:
        print("Usage: python simulate.py <strategy>")
        exit(1)

    strategy = argv[1]
    main(strategy)