from simulate import simulate
from evaluate.plot import plot_rs
from env import StudentAction, TeacherAction
import torch


def main():
    print("running random simulation")
    from policy.RandomPolicy import StudentPolicy, TeacherPolicy
    sπ = StudentPolicy()
    tπ = TeacherPolicy()
    l_random = simulate(35, 365, sπ, tπ)

    print("running q table simulation")
    from policy.FromQ import load_qpolicy
    sπ = load_qpolicy("model/50y-student-Q.pkl", StudentAction)
    tπ = load_qpolicy("model/50y-teacher-Q.pkl", TeacherAction)
    l_q = simulate(35, 365, sπ, tπ)

    print("running deep q, memoryless simulation")
    from policy.DeepQMemoryless.Student import StudentQ, StudentPolicy
    sQ = StudentQ(32)
    sQ.load_state_dict(torch.load("model/deep-q-memoryless-student.pt"))
    sπ = StudentPolicy(sQ)

    from policy.DeepQMemoryless.Teacher import TeacherQ, TeacherPolicy
    tQ = TeacherQ(32)
    tQ.load_state_dict(torch.load("model/deep-q-memoryless-teacher.pt"))
    tπ = TeacherPolicy(tQ)

    l_qdl_memoryless = simulate(35, 365, sπ, tπ)

    print("running deep q simulation")
    from policy.DeepQ.Teacher import TeacherQ, TeacherPolicy
    tQ = TeacherQ(16, 32)
    tQ.load_state_dict(torch.load("model/deep-q-teacher.pt"))
    tπ = TeacherPolicy(tQ, train=False)

    l_qdl = simulate(35, 365, sπ, tπ)

    plot_rs(
        [l_random, l_q, l_qdl_memoryless, l_qdl],
        [
            ("random policy", "random policy"),
            ("q-learned", "q-learned"),
            ("dqn (memoryless)", "dqn (memoryless)"),
            ("dqn (memoryless)", "dqn")
        ]
    )


if __name__ == '__main__':
    main()
