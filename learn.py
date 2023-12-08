"""
File: learn.py
--------------

This file implements the QLearning algorithm on a pandas dataframe.
"""
import pandas as pd
from plan.QLearning import sarsa, infer_fields
from policy.FromQ import QPolicy
from env import StudentAction, TeacherAction
from pickle import dump

DATA_NAME = "50y-1m"


def main():
    print("1. student")
    df = pd.read_csv(f"data/{DATA_NAME}/student-random.csv")
    fields = infer_fields(df)

    Q = sarsa(df, fields=fields, verbose=True)
    obs_ordering = [f[2:] for f in fields['o']]
    act_ordering = [f[2:] for f in fields['a']]
    dump(
        {"Q": Q, "obs_ordering": obs_ordering, "act_ordering": act_ordering},
        open(f"model/{DATA_NAME}-student-Q.pkl", "wb")
    )

    print("2. teacher")
    df = pd.read_csv(f"data/{DATA_NAME}/teacher-random.csv")
    fields = infer_fields(df)

    Q = sarsa(df, fields=fields, verbose=True)
    obs_ordering = [f[2:] for f in fields['o']]
    act_ordering = [f[2:] for f in fields['a']]
    dump(
        {"Q": Q, "obs_ordering": obs_ordering, "act_ordering": act_ordering},
        open(f"model/{DATA_NAME}-teacher-Q.pkl", "wb")
    )


if __name__ == "__main__":
    main()
