from simulate import simulate
from evaluate.plot import plot_rs
from env import StudentAction, TeacherAction
import torch


def main():
    from policy.RandomPolicy import StudentPolicy, TeacherPolicy
    sπ = StudentPolicy()
    tπ = TeacherPolicy()
    l_random = simulate(35, 365, sπ, tπ)

    from policy.FromQ import load_qpolicy
    sπ = load_qpolicy("model/50y-student-Q.pkl", StudentAction)
    tπ = load_qpolicy("model/50y-teacher-Q.pkl", TeacherAction)
    l_q = simulate(35, 365, sπ, tπ)
    # print(f"student took {sπ.num_rand} random actions, {sπ.num_q} q actions")
    # print(f"teacher took {tπ.num_rand} random actions, {tπ.num_q} q actions")

    from policy.DeepQMemoryless.Student import StudentQ, StudentPolicy
    sQ = StudentQ(32)
    sQ.load_state_dict(torch.load("model/deep-q-memoryless-student.pt"))
    sπ = StudentPolicy(sQ)

    from policy.DeepQMemoryless.Teacher import TeacherQ, TeacherPolicy
    tQ = TeacherQ(32)
    tQ.load_state_dict(torch.load("model/deep-q-memoryless-teacher.pt"))
    tπ = TeacherPolicy(tQ)

    l_q_deeplearned = simulate(35, 365, sπ, tπ)

    plot_rs(
        [l_random, l_q, l_q_deeplearned],
        [
            ("random policy", "random policy"),
            ("q-learned", "q-learned"),
            ("deep q-learned", "deep q-learned")
        ]
    )


if __name__ == '__main__':
    main()
