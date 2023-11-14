from sys import argv
from env import Classroom, Policy

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

    for t in range(d):
        # 1. student actions
        student_as = [sπ[o] for o in c.student_o]
        student_rs = c.student_step(student_as)
        
        # 2. teacher actions
        teacher_a = tπ[c.teacher_o]
        teacher_r = c.teacher_step(teacher_a)

        # 3. print results
        avg_sr = sum(student_rs) / len(student_rs)
        print(f"[day {t}] avg student reward: {avg_sr:.2f}, teacher reward: {teacher_r:.2f}")



def main(strategy):
    strategy = strategy.lower()
    if strategy == 'random':
        from policy.RandomPolicy import StudentPolicy, TeacherPolicy
    else:
        raise ValueError(f"Unrecognized strategy: {strategy}")
    
    sπ = StudentPolicy()
    tπ = TeacherPolicy()

    start_simulation(35, 30, sπ, tπ)


if __name__ == '__main__':
    if len(argv) != 2:
        print("Usage: python simulate.py <strategy>")
        exit(1)

    strategy = argv[1]
    main(strategy)