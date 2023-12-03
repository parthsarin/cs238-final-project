"""
File: learn.py
--------------

This file implements learning algorithms on a pandas dataframe.
"""
import pandas as pd
import argparse
from plan.QLearning import sarsa, infer_fields
from plan.QMDP import QMDP, infer_fields
from policy.FromQ import QPolicy
from env import StudentAction, TeacherAction
from pickle import dump

DATA_NAME = "50y-1m"

def main(algorithm):
    print("1. student")
    df = pd.read_csv(f"data/{DATA_NAME}/student-random.csv")
    fields = infer_fields(df)

    if algorithm == 'sarsa':
        Q = sarsa(df, fields=fields, verbose=True)
    elif algorithm == 'qmdp':
        Q = QMDP(df)

    obs_ordering = [f[2:] for f in fields['o']]
    act_ordering = [f[2:] for f in fields['a']]
    dump(
        {"Q": Q, "obs_ordering": obs_ordering, "act_ordering": act_ordering},
        open(f"model/{DATA_NAME}-student-{algorithm.upper()}.pkl", "wb")
    )

    print("2. teacher")
    df = pd.read_csv(f"data/{DATA_NAME}/teacher-random.csv")
    fields = infer_fields(df)

    if algorithm == 'sarsa':
        Q = sarsa(df, fields=fields, verbose=True)
    elif algorithm == 'qmdp':
        Q = QMDP(df)

    obs_ordering = [f[2:] for f in fields['o']]
    act_ordering = [f[2:] for f in fields['a']]
    dump(
        {"Q": Q, "obs_ordering": obs_ordering, "act_ordering": act_ordering},
        open(f"model/{DATA_NAME}-teacher-{algorithm.upper()}.pkl", "wb")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a learning algorithm on a DataFrame.")
    parser.add_argument("algorithm", choices=['sarsa', 'qmdp'], help="The learning algorithm to use (sarsa or qmdp)")
    args = parser.parse_args()
    main(args.algorithm)
