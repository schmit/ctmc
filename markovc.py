def create_uniform_rate_matrix(n):
    """
    Generate uniform choice transition rate matrix
    """
    Q = np.random.rand(n, n)
    Q = np.triu(Q - np.diag(np.diag(Q)))
    Q += np.tril(1-Q.T) - np.eye(n)
    return Q

def submatrix(A, indices):
    """
    Restrict A to a subset of indices
    """
    return A[indices,:][:,indices]

def embedded_jump_chain(Q):
    """
    Compute the embedded jump chain for a continuous time Markov Chain
    """
    P = np.divide(Q.T, np.sum(Q, 1)).T
    P -= np.diag(np.diag(P))
    return P

def equi_dtmc(P):
    """
    Compute equilibrium for discrete time Markov Chain
    """
    pi = np.real(sp.linalg.eig(P.T)[1])[:, 0]
    return pi / np.sum(pi)

def equi_ctmc(Q):
    """
    Compute equilibrium for continuous time Markov Chain
    """
    nu = np.sum(Q, 1)
    P = embedded_jump_chain(Q)
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
    rate = np.sum(qstate)
    weights = qstate / rate
    new_state = np.random.choice(n, p=weights)
    time = np.random.exponential(rate)
    return new_state, time

def transition_dtmc(P, state):
    """
    One step transition for discrete time markov chain
    """
    n, _ = Q.shape
    pstate = P[state, :]
    new_state = np.random.choice(n , p=pstate)
    return new_state
