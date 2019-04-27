import toolz

import operators as op


def generational(evals, initialize, pipeline, evaluate):
    """
     Defines the top level of an evolutionary algorithm with a non-overlapping population model: the population is
     completely replaced at each step with the offspring population.

    :param int evals: The number of fitness evaluation to run the algorithm for.  The EA will stop at the first generation that goes above this number, so it may execute a few extra evaluations if λ doesn't divide evently into `evals`.
    :param callable initialize: A callable that generates a brand new population when called.
    :param list pipeline: A pipeline of callables used to select parents and produce offspring.
    :param callable evaluate: A callable that evalutes the fitness of every individual in a population, and returns the population.
    :return: A generator that yields `(i, best)` for each generation, where `i` is time measured in the number of evaluations, and `best` is the best individual of that generation.


    To use this algorithm, you'll need to specify each of its components.  You'll usually want to use the
    :py:mod:`operators` module for selection and reproduction functions, and one of the domain-specific modules (like
    :py:mod:`real`) for the `initializer`:

    >>> import operators as op
    >>> import real
    >>> pop_size = 5
    >>> l = 5
    >>> ea = generational(evals=1000,
    ...                   initialize=real.random_initializer(pop_size, genome_length=l),
    ...                   pipeline=[
    ...                        op.tournament(n=pop_size),
    ...                        op.cloning,
    ...                        op.mutate_gaussian(prob=0.1, std=0.05)
    ...                   ],
    ...                   evaluate=op.evaluate(
    ...                       fitness_function=real.cosine_family(alpha=0.6,
    ...                                                           global_optima_counts=[5]*l,
    ...                                                           local_optima_counts=[5]*l)
    ...                   ))
    >>> ea # doctest:+ELLIPSIS
    <generator ...>

    The algorithm evaluates lazily when you query the generator:

    >>> print(*list(ea), sep='\\n') # doctest:+ELLIPSIS
    (10, [...])
    (15, [...])
    ...
    (1000, [...])

    """
    population = initialize()
    population = list(evaluate(population))

    i = len(population)
    while i < evals:
        # Create a generator that selects parents and creates offspring
        population = toolz.pipe(population, *pipeline)
        population = list(population)  # Execute it
        population = evaluate(population)
        i += len(population)

        yield (i, op.best(population))


def mu_plus_lambda(evals, initialize, parent_selector, pipeline, evaluate):
    """
    Defines the top level of an evolutionary algorithm with a (μ + λ)-style population model.

    :param int evals: The number of fitness evaluation to run the algorithm for.  The EA will stop at the first generation that goes above this number, so it may execute a few extra evaluations if λ doesn't divide evently into `evals`.
    :param callable initialize: A callable that generates a brand new population when called.
    :param callable parent_selector: A callable that selects μ individuals from the population.
    :param list pipeline: A pipeline of callables used to produce offspring.
    :param callable evaluate: A callable that evalutes the fitness of every individual in a population, and returns the population.
    :return: A generator that yields `(i, best)` for each generation, where `i` is time measured in the number of evaluations, and `best` is the best individual of that generation.


    To use this algorithm, you'll need to specify each of its components.  You'll usually want to use the
    :py:mod:`operators` module for selection and reproduction functions, and one of the domain-specific modules (like
    :py:mod:`real`) for the `initializer`:

    >>> import operators as op
    >>> import real
    >>> mu = 5
    >>> lambda_ = 10
    >>> l = 5
    >>> ea = mu_plus_lambda(evals=1000,
    ...                     initialize=real.random_initializer(mu + lambda_, genome_length=l),
    ...                     parent_selector=op.truncation(mu=mu),
    ...                     pipeline=[
    ...                          op.cloning(offspring_per_ind=int(lambda_/mu)),
    ...                          op.mutate_gaussian(prob=0.1, std=0.05)
    ...                     ],
    ...                     evaluate=op.evaluate(
    ...                         fitness_function=real.cosine_family(alpha=0.6,
    ...                                                             global_optima_counts=[5]*l,
    ...                                                             local_optima_counts=[5]*l)
    ...                     ))
    >>> ea # doctest:+ELLIPSIS
    <generator ...>

    The algorithm evaluates lazily when you query the generator:

    >>> print(*list(ea), sep='\\n') # doctest:+ELLIPSIS
    (25, [...])
    (35, [...])
    ...
    (1005, [...])


    """
    population = initialize()
    population = list(evaluate(population))

    i = len(population)
    while i < evals:
        # Selects parents
        parents = list(parent_selector(population))

        # Create a generator that creates offspring
        offspring = toolz.pipe(parents, *pipeline)
        offspring = list(offspring)  # Execute it

        population = parents + evaluate(offspring)
        i += len(offspring)

        yield (i, op.best(population))