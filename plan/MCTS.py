"""
File: MCTS.py
------------------

This file implements the Monte Carlo Tree Search algorithm on a pandas dataframe.
"""
import pandas as pd
from collections import defaultdict
from typing import Union
from tqdm import tqdm
from random import choice
import math import log

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0

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

def get_actions(state):
    # get the possible actions based on state (aka entire dataframe)
    # return [, , ...]
    pass

def take_action(state, action):
    # take action and transition to next state (aka entire dataframe)
    # return (next) state
    pass

def get_reward(state):
    # get reward based on state (aka entire dataframe)
    # return reward value
    pass

def select(node):
    # base case: return current node if leaf node aka no children
    if not node.children:
        return node
    # use UCB1 formula to balance exploitation and exploration
    selected_child = max(node.children, max_criteria=lambda child: child.value / child.visits + math.sqrt(2 * log(node.visits) / child.visits))
    # recursively traverse tree and find the best child
    return select(selected_child)

# GOAL: grow search tree by adding child node based on randomly chosen action that's valid for current state
def expand(node):
    # get possible actions from state
    actions = get_actions(node.state)

    # just return node if no possible actions
    if len(actions) == 0:
        return node

    # otherwise, randomly select an action
    action = choice(actions)
    # transition to next state with that action
    next_state = take_action(node.state, action)

    # use next state to make new child node, set parent to current node
    child = Node(next_state, parent=node)
    # add child to current node's list of children
    node.children.append(child)

    # return child for simulation
    return child

# GOAL: perform Monte Carlo simulations from an input node by randomly choosing actions for each simulation, updating the current state, and accumulating rewards
# Returns: average reward over all simulations ~ estimated reward for input node
def simulate(node):
    current_state = node.state
    total_reward = 0
    num_simulations = 8 # try changing!
    for i in range(num_simulations):
        # continue until no more possible actions from current state
        while True:
            actions = get_actions(current_state)
            if len(actions) == 0:
                break
            # choose random action from valid actions and update current state
            action = choice(actions)
            current_state = take_action(current_state, action)

            total_reward += get_reward(current_state)

    return total_reward/num_simulations

def backpropagate(node, reward):
    # loop from node to root by accessing parents
    while node is not None:
        node.visits += 1 # update visit counts for each node
        node.value += reward # add reward from simulated rollout
        node = node.parent

# GOAL: return the action taken to reach child state from parent state
def get_action_from_states(state1, state2):
    # get action that transitions from state1 to state2
    # ideally, state1 = parent and state2 = child
    pass

# GOAL: return the best action based on tree search results, prioritizing actions that have been explored most in simulations
def get_best_action(node):
    if len(node.children) == 0:
        return None
    # look at node's children -> select action associated with child node with highest visit count

    best_child = max(node.children, max_criteria=lambda child: child.visits)
    return get_action_from_states(node.state, best_child.state)

def mcts(df, iterations = 100):
    # initial state = entire data frame or is it just the first row?
    root_state = df
    root_node = Node(root_state)
    # or root_node = df.iloc[0]

    for i in range(iterations):
        selected_node = select(root_node)
        expanded_node = expand(selected_node)
        simulated_rollout = simulate(expanded_node)
        backpropagate(expanded_node, simulated_rollout)

    return get_best_action(root_node)

# best policy to try = mcts(df, iterations = 100)




