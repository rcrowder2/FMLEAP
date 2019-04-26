import numpy as np


def onemax(genome):
    """
    >>> genome = [0, 1, 1, 0, 0, 0, 1]
    >>> onemax(genome)
    3
    """
    return np.sum(genome)