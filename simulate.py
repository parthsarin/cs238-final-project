from sys import argv
from env import Classroom, Policy
from evaluate import Log
from evaluate.plot import plot_rs


def simulate(
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

    # record initial state
    l.record(-1)

    for t in range(d):
        # 1. student actions
        student_as = [sπ[o] for o in c.student_o]
        student_rs = c.student_step(student_as, t)

        # 2. teacher actions
        teacher_a = tπ[c.teacher_o]
        teacher_r = c.teacher_step(teacher_a, t)

        # 3. record results
        l.record(
            t,
            teacher_a=teacher_a,
            teacher_r=teacher_r,
            student_as=student_as,
            student_rs=student_rs
        )

    return l


def main():
    from policy.RandomPolicy import StudentPolicy, TeacherPolicy
    sπ = StudentPolicy()
    tπ = TeacherPolicy()
    l_random = simulate(35, 365, sπ, tπ)

    from policy.Always import StudentAlwaysWork, TeacherAlwaysGrade
    sπ = StudentAlwaysWork()
    tπ = TeacherAlwaysGrade()
    l_always_work = simulate(35, 365, sπ, tπ)

    from policy.Always import StudentAlwaysRest, TeacherAlwaysRest
    sπ = StudentAlwaysRest()
    tπ = TeacherAlwaysRest()
    l_always_rest = simulate(35, 365, sπ, tπ)

    plot_rs(
        [l_random, l_always_work, l_always_rest],
        [
            ("random policy", "random policy"),
            ("always work", "always grade"),
            ("always rest", "always rest")
        ]
    )


if __name__ == '__main__':
    main()
