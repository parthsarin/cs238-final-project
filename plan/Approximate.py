"""
File: Approximate.py
------------------

This file implements Approximate Q-Learning algorithms on a pandas dataframe.
"""
import pandas as pd
from collections import defaultdict
from typing import Union
from tqdm import tqdm
from random import choice
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LinearRegression
import numpy as np

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

def ks_helper(q_values, weights, bandwidth):
    """
    Helper function for kernel smoothing.

    PARAMS:
    q_values: np.array of Q-values to be smoothed
    weights: np.array of weights for each Q-value
    bandwidth: float for kernel density estimator

    RETURNS:
    smoothed_value: float
    """
    kde = KernelDensity(bandwidth=bandwidth)
    # fit kernel density estimator to input Q-values
    kde.fit(q_values.reshape(-1,1), sample_weight = weights)
    # use trained kernel density estimator to score query point: 0 (assuming we're smoothing Q-values around 0)
    smoothed_value = kde.score_samples([[0]])
    # score_samples is a one-element array, so extract the smoothed value
    return smoothed_value[0]
def kernel_smoothing(
    df: pd.DataFrame,
    num_simulations: int = 1000,
    num_samples: int = 100,
    lr: float = 1e-2,
    gamma: float = 0.95,
    Q: Union[dict, None] = None,
    fields: Union[dict, None] = None,
    replay_every: int = 10,
    verbose: bool = False,
):
    """
    Performs SARSA with kernel smoothing on a pandas dataframe.

    params
    ------
    df: pd.DataFrame
        dataframe containing the data
    num_simulations: int
        number of simulations to run
    num_samples: int
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
        how often to add a replay batch
    verbose: bool
        whether or not to print out the progress of the algorithm

    returns
    -------
    Q: dict
        the learned Q values
    """
    # setup the fields
    verify_inputs(df, num_simulations, num_samples, lr, gamma, Q, fields)
    if fields is None:
        fields = infer_fields(df)
    if Q is None:
        Q = defaultdict(float)

    # fill the null values with large numbers so they move to their own place
    # in the KDTree
    df = df.fillna(1e9)

    # verbose setup
    iterable = range(num_simulations)
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
            # sample num_samples rows from the dataframe
            sample = df.sample(num_samples)

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

        # track number of "cells" in Q dict that have not been updated ever
        null_count = sum(1 for value in Q.values() if value == 0.0)
        null_percentage = null_count/len(Q) * 100

        # if % of (o,a) pairs not updated ever is high, use approximate Q-learning
        if null_percentage > 30: # experiment with this threshold value
            # ------------------------------
            # ------ KERNEL SMOOTHING ------
            # ------------------------------
            # array of Q-values
            q_values = np.array(list(Q.values()))
            # get weights for each Q-value using Gaussian kernel function
            # Q-values closer to target (op, ap) pair get higher weights
            weights = np.exp(-np.sum((q_values - Q[op, ap])**2, axis=1) / (2 * 0.2**2)) # 0.2 = bandwidth
            # get smoothed Q-value
            smoothed_value = ks_helper(q_values, weights, 0.2) # 0.2 = bandwidth
            # update Q-value for current (o,a) pair
            Q[o,a] += lr * (r + gamma * Q[op, ap] - smoothed_value)

    return Q

def linear_interpolation(
    df: pd.DataFrame,
    num_simulations: int = 1000,
    num_samples: int = 100,
    lr: float = 1e-2,
    gamma: float = 0.95,
    Q: Union[dict, None] = None,
    fields: Union[dict, None] = None,
    replay_every: int = 10,
    verbose: bool = False,
):
    """
    Performs SARSA with linear interpolation on a pandas dataframe.
    """

    # setup the fields
    verify_inputs(df, num_simulations, num_samples, lr, gamma, Q, fields)
    if fields is None:
        fields = infer_fields(df)
    if Q is None:
        Q = defaultdict(float)

    # fill the null values with large numbers so they move to their own place
    # in the KDTree
    df = df.fillna(1e9)

    # verbose setup
    iterable = range(num_simulations)
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
            # sample num_samples rows from the dataframe
            sample = df.sample(num_samples)

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

        # track number of "cells" in Q dict that have not been updated ever
        null_count = sum(1 for value in Q.values() if value == 0.0)
        null_percentage = null_count/len(Q) * 100

        # if % of (o,a) pairs not updated ever is high, use approximate Q-learning
        if null_percentage > 30: # experiment with this threshold value
            # ------------------------------
            # ---- LINEAR INTERPOLATION ----
            # ------------------------------

            # array of all (o,a) pairs
            X = np.array(list(Q.keys()))
            # array of all Q-values
            Y = np.array(list(Q.values()))
            # train linear regression model using features: X,Y -> predict Q values based on (o,a) pairs
            lr_model = LinearRegression()
            lr_model.fit(X,Y)
            # use linear regression model to predict Q-value for current (o,a) pair
            interpolated_value = lr.model.predict(np.array([o,a]).reshape(1,-1))
            # update Q value for current (o,a) pair
            Q[o, a] += lr * (r + gamma * Q[op, ap] - interpolated_value)

    return Q
