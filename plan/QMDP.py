import numpy as np

class POMDP:
    def __init__(self, s, a, o, t, z, r, gamma):
        self.s = s  # states
        self.a = a  # actions
        self.o = o  # observations
        self.t = t  # transition function
        self.z = z  # observation function
        self.r = r  # reward function
        self.gamma = gamma  # discount factor

class AlphaVectorPolicy:
    def __init__(self, p, r, a):
        self.p = p  # POMDP problem
        self.r = r  # alpha vectors
        self.a = a  # actions associated with alpha vectors

def utility(pi, b):
    return max(np.dot(alpha, b) for alpha in pi.r)

def choose_action(pi, b):
    i = np.argmax([np.dot(alpha, b) for alpha in pi.r])
    return pi.a[i]

class QMDP:
    def __init__(self, k_max):
        self.k_max = k_max  # maximum number of iterations

def solve(m: QMDP, p: POMDP):
    r = [zeros(len(p.s)) for a in p.a]
    r = alphavector_iteration(m, p, r)
    return AlphaVectorPolicy(p, r, p.a)

def update(p: POMDP, m: QMDP, r):
    s, a, r, t, gamma = p.s, p.a, p.r, p.t, p.gamma
    r_prime = [[r[s, a] + gamma * sum(t(s, a, s_prime) * max(r_prime[j] for r_prime in r)
               for j, s_prime in enumerate(s)) for s in s] for a in a]
    return r_prime

def alphavector_iteration(p, m, r):
    for k in range(1, m.k_max + 1):
        r = update(p, m, r)
    return r

def zeros(length):
    return [0] * length

