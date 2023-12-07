"""
File: learn.py
--------------

This file implements the QLearning algorithm on a pandas dataframe.
"""
# from policy.DeepQMemoryless.Student import StudentQ, StudentPolicy
from policy.RandomPolicy import StudentPolicy
from policy.DeepQMemoryless.Teacher import TeacherQ, TeacherPolicy

from env import Classroom

import torch

# sQ = StudentQ(32)
# sQ.load_state_dict(torch.load("model/deep-q-memoryless-student.pt"))

tQ = TeacherQ(32)
opt = torch.optim.Adam(tQ.parameters(), lr=0.001)


def main():
    # run a simulation of the first 14 days with 35 students and then restart,
    # keep track of the loss
    sπ = StudentPolicy()
    tπ = TeacherPolicy(tQ)
    c = Classroom(35)
    loss = 0

    for epoch in range(100_000):
        # reset the classroom
        c = Classroom(35)

        if epoch < 500:
            horizon = 5
        elif epoch < 1000:
            horizon = 14
        else:
            horizon = 100

        # run the simulation
        for t in range(horizon):
            # student actions
            student_os = [h[-1][0] for h in c.student_h]
            student_as = [sπ[h] for h in c.student_h]
            student_rs = c.student_step(student_as, t)

            # teacher action
            teacher_o = c.teacher_h[-1][0]
            teacher_a = tπ[c.teacher_h]
            teacher_r = c.teacher_step(teacher_a, t)
            teacher_op = c.teacher_h[-1][0]

            # update the policy
            loss += tπ.loss(teacher_o, teacher_a, teacher_r, teacher_op)

        if epoch % 5 == 0:
            # update the policy
            opt.zero_grad()
            loss.backward()
            opt.step()

            # print the loss
            print(f"[epoch {epoch}] loss = {loss.item()}")

            # reset the loss
            loss = 0

            # write the model
            torch.save(tQ.state_dict(), "model/deep-q-memoryless-teacher.pt")


if __name__ == "__main__":
    main()
