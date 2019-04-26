import numpy as np
from toolz import curry


@curry
def evaluate(population, fitness_function):
    """
    >>> from boolean import onemax
    >>> from individual import Individual
    >>> pop = [Individual(genome=x) for x in [[0, 1, 1, 0, 0, 0, 1], [0, 1, 1, 1, 1, 0, 1]]]
    >>> pop = evaluate(pop, fitness_function=onemax)
    >>> pop[0].fitness
    3

    >>> pop[1].fitness
    5
    """
    for ind in population:
        if ind.fitness is None:
            ind.fitness = fitness_function(ind.genome)
    return population


@curry
def mutate_bitflip(population, prob):
    """
    >>> from individual import Individual
    >>> population = [Individual(genome=[1, 0, 1, 1, 0])]
    >>> always = mutate_bitflip(prob=1.0)
    >>> list(always(population))
    [[0, 1, 0, 0, 1]]

    Individuals are modified in place:
    >>> population
    [[0, 1, 0, 0, 1]]

    >>> population = [Individual(genome=[1, 0, 1, 1, 0])]
    >>> never = mutate_bitflip(prob=0.0)
    >>> list(never(population))
    [[1, 0, 1, 1, 0]]
    """
    def flip(x):
        if np.random.uniform() < prob:
            return 0 if x == 1 else 1
        else:
            return x

    for ind in population:
        ind.genome = [flip(x) for x in ind.genome]
        ind.fitness = None
        yield ind


@curry
def mutate_gaussian(population, prob, std):
    def add_gauss(x):
        if np.random.uniform() < prob:
            return x + np.random.normal()*std
        else:
            return x

    for ind in population:
        ind.genome = [add_gauss(x) for x in ind.genome]
        ind.fitness = None
        yield ind


@curry
def truncation(population, mu):
    """
    Returns the `mu` individuals with the best fitness.

    For example, say we have a population of 10 individuals with the following fitnesses:

    >>> from individual import Individual
    >>> fitnesses = [0.12473057, 0.74763715, 0.6497458 , 0.36178902, 0.41318757, 0.69130493, 0.67464942, 0.14895497, 0.15406642, 0.31307095]
    >>> population = [Individual(genome=[i], fitness=f) for i, f in zip(range(10), fitnesses)]

    The three highest-fitness individuals are are the indices 1, 5, and 6:

    >>> list(truncation(population, 3))
    [[1], [5], [6]]

    """
    inds = list(sorted(list(population), key=lambda x: x.fitness, reverse=True))
    for ind in inds[0:mu]:
        yield ind


def best(population):
    """
    >>> from individual import Individual
    >>> pop = [Individual(genome=[1, 0, 1, 1, 0], fitness=3), \
               Individual(genome=[0, 0, 1, 0, 0], fitness=1), \
               Individual(genome=[0, 1, 1, 1, 1], fitness=4), \
               Individual(genome=[1, 0, 0, 0, 1], fitness=2)]
    >>> best(pop)
    [0, 1, 1, 1, 1]
    """
    return max(population, key=lambda x: x.fitness)


@curry
def cloning(population, offspring_per_ind=1):
    """
    >>> pop = [[1, 2], [3, 4], [5, 6]]
    >>> new_pop = list(cloning(pop))
    >>> new_pop
    [[1, 2], [3, 4], [5, 6]]

    If we edit individuals in the original, new_pop shouldn't change:

    >>> pop[0][1] = 7
    >>> pop[2][0] = 0
    >>> pop
    [[1, 7], [3, 4], [0, 6]]

    >>> new_pop
    [[1, 2], [3, 4], [5, 6]]

    If we set `offspring_per_ind`, we can create bigger populations:

    >>> pop = [[1, 2], [3, 4], [5, 6]]
    >>> new_pop = list(cloning(pop, offspring_per_ind=3))
    >>> new_pop
    [[1, 2], [1, 2], [1, 2], [3, 4], [3, 4], [3, 4], [5, 6], [5, 6], [5, 6]]

    """
    assert(population is not None)
    assert(offspring_per_ind > 0)

    for ind in population:
        for i in range(offspring_per_ind):
            yield ind.copy()