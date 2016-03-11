import numpy as np

def get_subset(n, pinc=0.2):
    """
    Generate a subset of n with at least 2 elements

    Args:
        n: number of elements
        pinc: probability of including extra elements

    Returns:
        [elements]
    """
    assert n > 1
    base_set = list(np.random.choice(n, 2, replace=False))
    additional_set = [x for x in range(n) if random.random() < pinc]
    total_set = set(base_set + additional_set)
    return list(total_set)

def pick_winner(Q, choice_set):
    """
    Pick a winner among states according to PCMC model with transition
    rate matrix Q

    Args:
        Q: Transition rate matrix
        states: choice set
    """
    return np.random.choice(len(choice_set), p=equi_ctmc(submatrix(Q, choice_set)))

def get_observation(Q, pinc=0.2):
    """
    Generate a single observed choice

    Args:
        Q: Transition rate matrix
        pinc: probability of including states (besides 2 random states)

    Returns:
        (winner, choice set)
    """
    n, _ = Q.shape
    subset = get_subset(n, pinc)
    winner = pick_winner(Q, subset)
    return winner, subset

def gen_data(Q, n=10, pinc=0.2):
    """
    Generate n samples of observed choices generated from transition matrix Q

    Args:
        Q: transition rate matrix
        n: number of observations
        pinc: probability of including states (besides 2 random states)

    Returns:
        [(winner, choice set)]
    """
    return [get_observation(Q, pinc) for _ in range(n)]
