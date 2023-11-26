from simulate import simulate
from evaluate.plot import plot_rs
from env import StudentAction, TeacherAction


def main():
    from policy.RandomPolicy import StudentPolicy, TeacherPolicy
    sπ = StudentPolicy()
    tπ = TeacherPolicy()
    l_random = simulate(35, 365, sπ, tπ)

    from policy.FromQ import load_qpolicy
    sπ = load_qpolicy("model/50y-student-Q.pkl", StudentAction)
    tπ = load_qpolicy("model/50y-teacher-Q.pkl", TeacherAction)
    l_q = simulate(35, 365, sπ, tπ)

    print(f"student took {sπ.num_rand} random actions, {sπ.num_q} q actions")
    print(f"teacher took {tπ.num_rand} random actions, {tπ.num_q} q actions")

    plot_rs(
        [l_random, l_q],
        [
            ("random policy", "random policy"),
            ("q-learned", "q-learned")
        ]
    )


if __name__ == '__main__':
    main()
