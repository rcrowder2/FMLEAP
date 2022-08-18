#!/usr/bin/env python3
"""
    Used to initialize binary sequences
"""
import numpy as np

from leap_ec.individual import Individual
from leap_ec.real_rep.ops import apply_hard_bounds

##############################
# Closure create_binary_sequence
##############################
def create_binary_sequence(length):
    """
    A closure for initializing a binary sequences for binary genomes.

    :param length: how many genes?

    :return: a function that, when called, generates a binary vector of given
        length

    E.g., can be used for `Individual.create_population`

    >>> from leap_ec.decoder import IdentityDecoder
    >>> from . problems import MaxOnes
    >>> population = Individual.create_population(10, create_binary_sequence(length=10),
    ...                                           decoder=IdentityDecoder(),
    ...                                           problem=MaxOnes())

    """

    def create():
        return np.random.choice([0, 1], size=(length,))

    return create

def create_bounded_sequence(length,bounds):
    
    def create():
        # genome = np.random.choice([bounds[0],bounds[3]], size=(length,))
        
        # for i,gene in enumerate(genome):
        #     if gene == bounds[0]:
        #         genome[i] = np.random.uniform(bounds[0],bounds[1])
        #     else:
        #         genome[i] = np.random.uniform(bounds[2],bounds[3])
        # # genome = np.random.normal(genome,0.03)
        
        # return apply_hard_bounds(genome, (bounds[0],bounds[3]))
        return np.random.choice([bounds[0],bounds[1]], size=(length,))
        
    return create