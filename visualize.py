from simulate import simulate
from evaluate.plot import plot_rs
from env import StudentAction, TeacherAction
import torch
import pandas as pd


def main():
    print("running random simulation")
    from policy.RandomPolicy import StudentPolicy, TeacherPolicy
    sπ = StudentPolicy()
    tπ = TeacherPolicy()
    l_random = simulate(35, 365, sπ, tπ)

    # print("running q table simulation")
    # from policy.FromQ import load_qpolicy
    # sπ = load_qpolicy("model/50y-student-Q.pkl", StudentAction)
    # tπ = load_qpolicy("model/50y-teacher-Q.pkl", TeacherAction)
    # l_q = simulate(35, 365, sπ, tπ)

    # print("running q + li simulation")
    # from policy.FromQ import load_qpolicy
    # sπ = load_qpolicy("model/50y-1m-student-Q-li.pkl", StudentAction)
    # tπ = load_qpolicy("model/50y-1m-teacher-Q-li.pkl", TeacherAction)
    # l_q_li = simulate(35, 365, sπ, tπ)

    # print("running epsilon-greedy simulation")
    # from policy.FromEpsilonGreedy import load_epsilon_greedy_policy
    # sπ = load_epsilon_greedy_policy("model/50y-student-Q.pkl", StudentAction)
    # tπ = load_epsilon_greedy_policy("model/50y-teacher-Q.pkl", TeacherAction)
    # l_e = simulate(35, 365, sπ, tπ)

    # print("running ucb1 simulation")
    # from policy.FromUCB1 import load_ucb1_policy
    # sπ = load_ucb1_policy("model/50y-student-Q.pkl", StudentAction)
    # tπ = load_ucb1_policy("model/50y-teacher-Q.pkl", TeacherAction)
    # l_ucb1 = simulate(35, 365, sπ, tπ)

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
    tQ = TeacherQ(3, 128)
    tQ.load_state_dict(torch.load("model/deep-q-teacher-memoryless-2.pt"))
    tπ = TeacherPolicy(tQ, train=False)

    from policy.DeepQ.Student import StudentQ, StudentPolicy
    sQ = StudentQ(3, 32)
    sQ.load_state_dict(torch.load("model/deep-q-student-memoryless-2.pt"))
    sπ = StudentPolicy(sQ, train=False)

    l_qdl = simulate(35, 365, sπ, tπ)

    plot_rs(
        [
            l_random,
            # l_q,
            # l_q_li,
            # l_e,
            # l_ucb1,
            l_qdl_memoryless,
            l_qdl
        ],
        [
            ("random policy", "random policy"),
            # ("q-learning", "q-learning"),
            # ("q + linear interpolation", "q + linear interpolation"),
            # ("q + ε-greedy", "q + ε-greedy"),
            # ("q + ucb1", "q + ucb1"),
            ("dqn (memoryless)", "dqn (memoryless)"),
            ("dqn", "dqn")
        ]
    )


if __name__ == '__main__':
    main()
