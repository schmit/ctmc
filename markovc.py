import numpy as np
import scipy as sp

import scipy.linalg

def create_rate_matrix_from_vector(x):
    """
    Create arbitrary rate matrix from a vector x
    where the entries are the (i,j)th entries of the rate matrix
    for i > j
    """
    # find size of Q
    m = len(x)
    n = int(1/2 + np.sqrt(1/4 + 2 * m)+0.5)

    # set top part of Q
    Q = np.zeros((n, n))
    Q[np.triu_indices(n, 1)] = x

    # set bottom part of Q
    Q += np.tril(1-Q.T, -1)

    # fix diagonal
    Q -= np.diag(np.sum(Q, 1))

    return Q

def create_uniform_rate_matrix(n):
    """
    Generate uniform random choice transition rate matrix
    """
    m = n * (n-1) / 2
    x = np.random.rand(m)

    return create_rate_matrix_from_vector(x)

def submatrix(A, indices):
    """
    Restrict A to a subset of indices
    """
    return A[indices,:][:,indices]

def embedded_jump_chain(Q):
    """
    Compute the embedded jump chain for a continuous time Markov Chain
    """
    P  = np.divide(Q.T, -np.diag(Q)).T
    P -= np.diag(np.diag(P))
    return P

def equi_dtmc(P):
    """
    Compute equilibrium for discrete time Markov Chain
    """
    pi = np.real(sp.linalg.eig(P.T)[1])[:, 0]
    return pi / np.sum(pi)

def equi_deriv_ctmc(Q,P,pi,coord):
    """
    Compute derivative of equilibrium for continuous time Markov chain
    """
    i,j      = coord
    n,_      = Q.shape

    # Note: Should build this as a sparse matrix later
    dP = np.zeros((n,n))
    dP[:,i] = -Q[:,i] / Q[i,i]
    dP[:,j] =  Q[:,j] / Q[j,j]
    dP[i,j] = -(Q[j,j] + P[i,j]) / Q[j,j] ** 2
    dP[j,i] =  (Q[i,i] + P[j,i]) / Q[i,i] ** 2

    dpi = np.linalg.pinv(np.eye(n) - P) @ (dP @ pi)

    v = -np.diag(Q)

    numerator   = pi / v
    denominator = np.sum(numerator)

    dnum    = dpi / v
    dnum[i] = (v[i] * dpi[i] + pi[i]) / v[i] ** 2
    dnum[j] = (v[j] * dpi[j] - pi[j]) / v[j] ** 2
    dden    = np.sum(dnum)

    dqi = (denominator * dnum - numerator * dden) / denominator ** 2

    return dqi


def equi_ctmc(Q):
    """
    Compute equilibrium for continuous time Markov Chain
    """
    nu = np.sum(Q, 1)
    P  = embedded_jump_chain(Q)
    pi = equi_dtmc(P)
    qi = pi / nu
    return qi / np.sum(qi)

def transition_ctmc(Q, state):
    """
    One step transition for continuous time markov chain

    Returns:
        (new_state, time)
    """
    n, _ = Q.shape
    qstate = Q[state, :]
    rate = -qstate[state]
    weights = qstate / rate
    weights[state] = 0
    new_state = np.random.choice(n, p=weights)
    time = np.random.exponential(rate)
    return new_state, time

def transition_dtmc(P, state):
    """
    One step transition for discrete time markov chain
    """
    n, _ = P.shape
    pstate = P[state, :]
    new_state = np.random.choice(n , p=pstate)
    return new_state
