"""
File: QLearning.py
------------------

This file implements the QLearning algorithm on a pandas dataframe.
"""
import pandas as pd
from collections import defaultdict
from typing import Union
from tqdm import tqdm
from random import choice


def infer_fields(df: pd.DataFrame):
    c = df.columns
    return {
        "o": [c for c in c if c.startswith("o_")],
        "a": [c for c in c if c.startswith("a_")],
        "r": [c for c in c if c.startswith("r")],
        "op": [c for c in c if c.startswith("op_")],
        "ap": [c for c in c if c.startswith("ap_")],
    }


def verify_inputs(df, m, d, lr, gamma, Q, fields):
    assert 0 <= lr <= 1, "learning rate (alpha) must be between 0 and 1"
    assert 0 <= gamma <= 1, "discount factor (gamma) must be between 0 and 1"

    if fields is not None:
        assert all(
            [k in fields for k in ("o", "a", "r", "op", "ap")]
        ), "fields must contain keys: o, a, r, op, ap"


def sarsa(
    df: pd.DataFrame,
    m: int = 1000,
    d: int = 100,
    lr: float = 1e-2,
    gamma: float = 0.95,
    Q: Union[dict, None] = None,
    fields: Union[dict, None] = None,
    replay_every: int = 10,
    verbose: bool = False,
):
    """
    Performs SARSA on a pandas dataframe.

    params
    ------
    df: pd.DataFrame
        dataframe containing the data
    m: int
        number of simulations to run
    d: int
        number of samples each simulation
    lr: float
        learning rate
    gamma: float
        discount factor
    Q: dict
        initial Q values
    fields: dict
        dictionary containing the column names for the observation, action,
        reward, next observation, and next action
    replay_every: int
        how often to add an replay batch
    verbose: bool
        whether or not to print out the progress of the algorithm

    returns
    -------
    Q: dict
        the learned Q values
    """
    # setup the fields
    verify_inputs(df, m, d, lr, gamma, Q, fields)
    if fields is None:
        fields = infer_fields(df)
    if Q is None:
        Q = defaultdict(float)

    # fill the null values with large numbers so they move to their own place
    # in the KDTree
    df = df.fillna(1e9)

    # verbose setup
    iterable = range(m)
    if verbose:
        iterable = tqdm(iterable)

    # run the algorithm
    batches = []
    for batch in iterable:
        # is this a replay batch?
        if batch % replay_every == 0 and batches:
            # replay an old batch
            replay_batch = choice(batches)
            sample = df.loc[replay_batch]

        else:
            # sample d rows from the dataframe
            sample = df.sample(d)

            # add the indexes to the list of batches
            batches.append(sample.index)

        # update the Q values
        for _, row in sample.iterrows():
            # get the observation, action, reward, next observation, and next action
            o = tuple(row[fields["o"]].to_numpy())
            a = tuple(row[fields["a"]].to_numpy())
            r = row[fields["r"]].item()
            op = tuple(row[fields["op"]].to_numpy())
            ap = tuple(row[fields["ap"]].to_numpy())

            # update the Q value
            Q[o, a] += lr * (r + gamma * Q[op, ap] - Q[o, a])

    return Q
