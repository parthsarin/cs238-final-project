"""
File: learn.py
--------------

This file implements the QLearning algorithm on a pandas dataframe.
"""
# from policy.RandomPolicy import StudentPolicy, TeacherPolicy
# from policy.DeepQMemoryless.Student import StudentQ, StudentPolicy
# from policy.DeepQMemoryless.Teacher import TeacherQ, TeacherPolicy
from policy.DeepQ.Teacher import TeacherQ, TeacherPolicy
from policy.DeepQ.Student import StudentQ, StudentPolicy

from env import Classroom

import torch

sQ = StudentQ(5, 32)
s_opt = torch.optim.Adam(sQ.parameters(), lr=0.001)
# sQ.load_state_dict(torch.load("model/deep-q-student.pt"))

tQ = TeacherQ(5, 64)
t_opt = torch.optim.Adam(tQ.parameters(), lr=0.001)
# tQ.load_state_dict(torch.load("model/deep-q-teacher.pt"))


def main():
    # run a simulation of the first 14 days with 35 students and then restart,
    # keep track of the loss
    sπ = StudentPolicy(sQ)
    tπ = TeacherPolicy(tQ)
    c = Classroom(35)
    s_loss = t_loss = 0

    for epoch in range(100_000):
        # reset the classroom
        c = Classroom(35)

        if epoch < 100:
            horizon = 5
        elif epoch < 500:
            # ramp up to 21
            horizon = 5 + (epoch - 100) // 21
            sπ.eps = tπ.eps = 0.3
        else:
            horizon = 21
            sπ.eps = tπ.eps = 0.5

        # run the simulation
        for t in range(horizon):
            # student actions
            student_as = []
            student_qs = []
            for h in c.student_h:
                a, q = sπ[h]
                # a = sπ[h]
                student_as.append(a)
                student_qs.append(q)
            student_rs = c.student_step(student_as, t)

            # teacher action
            teacher_a, q_vals = tπ[c.teacher_h]
            # teacher_a = tπ[c.teacher_h]
            teacher_r = c.teacher_step(teacher_a, t)

            # update the policy
            student_hs = [h for h in c.student_h]
            for (q, a, r, h) in zip(student_qs, student_as, student_rs, student_hs):
                # for (a, r, h) in zip(student_as, student_rs, student_hs):
                #     o = h[-2][0]
                #     op = h[-1][0]
                s_loss += sπ.loss(q, a, r, h)
                # s_loss += sπ.loss(o, a, r, op)
            t_loss += tπ.loss(q_vals, teacher_a, teacher_r, c.teacher_h)
            # o = c.teacher_h[-2][0]
            # op = c.teacher_h[-1][0]
            # t_loss += tπ.loss(o, teacher_a, teacher_r, op)

        # update the student policy
        s_opt.zero_grad()
        s_loss.backward()
        s_opt.step()

        # update the teacher policy
        t_opt.zero_grad()
        t_loss.backward()
        t_opt.step()

        # print the loss
        print(
            f"[epoch {epoch}] student loss = {round(s_loss.item(), 2)}, teacher loss = {round(t_loss.item(), 2)}"
        )

        # reset the loss
        s_loss = t_loss = 0

        # write the model
        torch.save(sQ.state_dict(), "model/deep-q-student-2.pt")
        torch.save(tQ.state_dict(), "model/deep-q-teacher-2.pt")


if __name__ == "__main__":
    main()
