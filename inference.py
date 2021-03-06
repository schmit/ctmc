import numpy as np
from markovc import *

def nll(x, data):
    """
    Negative log-likelihood of x given data
    """
    Q = create_rate_matrix_from_vector(x)
    return sum(nll_term(Q, winner, choices, count) 
               for (winner, choices), count in data.items())

def nll_term(Q, winner, choices, count):
    """
    Single term of log-likelihood of xhat
    """

    Qs = subchain(Q, choices)

    winner_idx = [idx for idx, val in enumerate(choices) if winner==val][0]

    qiS = equi_ctmc(Qs)[winner_idx]
    return - count * np.log(np.abs(qiS))


