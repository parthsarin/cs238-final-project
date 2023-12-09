"""
File: learn.py
--------------

This file implements the QLearning algorithm on a pandas dataframe.
"""
from policy.RandomPolicy import StudentPolicy, TeacherPolicy
# from policy.DeepQ.Teacher import TeacherQ, TeacherPolicy
from policy.DeepQ.Student import StudentQ, StudentPolicy

from env import Classroom

import torch

sQ = StudentQ(16, 32)
opt = torch.optim.Adam(sQ.parameters(), lr=0.001)
# sQ.load_state_dict(torch.load("model/deep-q-memoryless-student.pt"))

# tQ = TeacherQ(16, 128)
# opt = torch.optim.Adam(tQ.parameters(), lr=0.001)


def main():
    # run a simulation of the first 14 days with 35 students and then restart,
    # keep track of the loss
    sπ = StudentPolicy(sQ)
    tπ = TeacherPolicy()
    c = Classroom(35)
    loss = 0

    for epoch in range(100_000):
        # reset the classroom
        c = Classroom(35)

        # run the simulation
        for t in range(5):
            # student actions
            student_as = []
            student_qs = []
            for h in c.student_h:
                a, q = sπ[h]
                student_as.append(a)
                student_qs.append(q)
            student_rs = c.student_step(student_as, t)

            # teacher action
            # teacher_a, q_vals = tπ[c.teacher_h]
            teacher_a = tπ[c.teacher_h]
            teacher_r = c.teacher_step(teacher_a, t)

            # update the policy
            student_hs = [h for h in c.student_h]
            for (q, a, r, h) in zip(student_qs, student_as, student_rs, student_hs):
                loss += sπ.loss(q, a, r, h)
            # loss += tπ.loss(q_vals, teacher_a, teacher_r, c.teacher_h)

        # update the policy
        opt.zero_grad()
        loss.backward()
        opt.step()

        # print the loss
        print(f"[epoch {epoch}] loss = {loss.item()}")

        # reset the loss
        loss = 0

        # write the model
        torch.save(sQ.state_dict(), "model/deep-q-student.pt")


if __name__ == "__main__":
    main()
