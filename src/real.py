import numpy as np

from individual import Individual


def random_initializer(pop_size, genome_length, gene_min=0.0, gene_max=1.0):
    """
    >>> pop = random_initializer(10, 5)
    >>> len(pop())
    10
    """
    def f():
        return [Individual(np.random.uniform(gene_min, gene_max, genome_length)) for _ in range(pop_size)]
    return f


def cosine_family(alpha, global_optima_counts, local_optima_counts):
    dimensions = len(global_optima_counts)
    assert(len(local_optima_counts) == dimensions)
    global_optima_counts = np.array(global_optima_counts)
    local_optima_counts = np.array(local_optima_counts)

    def f(genome):
        genome = np.array(genome)
        term1 = -np.cos((global_optima_counts - 1) * 2 * np.pi * genome)
        term2 = - alpha * np.cos((global_optima_counts - 1) * 2 * np.pi * local_optima_counts * genome)
        value = np.sum(term1 + term2)/(2*dimensions)
        # We modify the original function to make it a maximization problem
        # and so that the global optima are scaled to always have a fitness of 1
        return -2/(alpha + 1) * value
    return f
