import numpy as np
import pandas as pd

from env.Student import StudentAction, StudentState
from env.Teacher import TeacherAction, TeacherState

"""
This should take in the data from the student or teacher and output an AlphaVectorPolicy
"""
student_state_space = [
        StudentState(mh, prod, g, ft, na, tw, [])
        for mh in (-1, -0.5, 0, .5, 1)
        for prod in (0, 0.25, 0.5, 0.75, 1)
        for g in ([0] * 5, [0.25] * 5, [0.5] * 5, [0.75] * 5, [1] * 5)
        for na in range(15)
        for ft in range(8)
        for tw in range(37)
]

student_action_space = [StudentAction(True, None, None)] + [
        StudentAction(False, w, 1 - w)
        for w in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
]

teacher_state_space = [ 
    TeacherState(mh, prod, g, ft, na)
    for mh in (-1, -0.5, 0, .5, 1)
    for prod in (0, 0.25, 0.5, 0.75, 1)
    for g in (0, 0.25, 0.5, 0.75, 1)
    for na in range(15)
    for ft in range(8)
]

teacher_action_space = [
    TeacherAction(r, g, pd)
    for r in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
    for g in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
    for pd in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
    if r + g + pd == 1
]

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


class AlphaVectorPolicy:
    def __init__(self, p: POMDP, r, a):
        self.p = p  # POMDP problem
        self.r = r  # alpha vectors
        self.a = a  # actions associated with alpha vectors

class QMDP:
    def __init__(self, k_max):
        self.k_max = k_max  # maximum number of iterations

def qmdp(m: QMDP, agent: str):
    """
    Solves the POMDP using the QMDP algorithm.
    Parameters:
        p -- the POMDP problem
        m -- the QMDP parameters
    """
    def zeros(length):
        return [0] * length

    def update(m: QMDP, r):
        """
        Updates the alpha vectors using the QMDP algorithm.
        Parameters:
            p -- the POMDP problem
            m -- the QMDP parameters
            r -- the current alpha vectors
        """
        s, a, r_func, t, gamma = p.s, p.a, p.r, p.t, p.gamma
        r_prime = [[r_func[s, a] + gamma * sum(t(s, a, s_prime) * max(r_prime[j] for r_prime in r)
                   for j, s_prime in enumerate(s)) for s in s] for a in a]
        return r_prime

    def alphavector_iteration(m: QMDP, r):
        """
        Performs the alpha vector iteration algorithm.
        Parameters:
            p -- the POMDP problem
            m -- the QMDP parameters
            r -- the current alpha vectors
        """
        for _ in range(1, m.k_max + 1):
            r = update(m, r)
        return r

    # Main logic of qmdp
    # check whether agent is student or teacher and set up state space and action space accordingly 
    if agent == "student":
        state_space = student_state_space
        action_space = student_action_space
    elif agent == "teacher":
        state_space = teacher_state_space
        action_space = teacher_action_space
    else:
        raise ValueError("Agent must be either student or teacher")
    
    r = [zeros(len(state_space)) for _ in len(action_space)]
    r = alphavector_iteration(m, r)
    return AlphaVectorPolicy(p, r, p.a)
    

def utility(pi, b):
    return max(np.dot(alpha, b) for alpha in pi.r)

def choose_action(pi, b):
    i = np.argmax([np.dot(alpha, b) for alpha in pi.r])
    return pi.a[i]
