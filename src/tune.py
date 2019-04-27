import numpy as np


def tune_random(algorithm, metric, samples, tests_per_sample=5, **kwargs):
    """
    Tune the real-valued parameters of an algorithm using random sampling.

    :param callable algorithm: The algorithm to tune, with any fixed parameters pre-set, so that it only takes variable a parameters as arguments.
    :param callable metric: A metric to evaluate the performance of algorithm configurations.  This should take the output of `algorithm` as its input.
    :param int samples: The number of random configurations to evaluate.
    :param int samples: The number of runs over which the score of each configuration will be averaged.
    :param kwargs: Named parameters to be tuned, each assigned a tuple of the form (min, max).
    :return: A generator yielding `(score, configuration)` pairs for the best-so-far configuration.

    For example, if we want to tune the `evolve()` algorithm, leaving its μ and λ parameters fixed.  We can use
    `functools.partial` to fix the parameters we don't want to tune:

    >>> from evolve import generational
    >>> import real
    >>> import operators as op
    >>> from toolz import curry
    >>>
    >>> @curry
    ... def my_algorithm(pop_size, mutation_prob, mutation_std):
    ...     return generational(evals=100,
    ...                         initialize=real.random_initializer(pop_size, genome_length=10),
    ...                         pipeline=[
    ...                              op.tournament(n=pop_size),
    ...                              op.cloning,
    ...                              op.mutate_gaussian(prob=mutation_prob, std=mutation_std)
    ...                         ],
    ...                         evaluate=op.evaluate(
    ...                             fitness_function=real.cosine_family(alpha=0.6,
    ...                                                                 global_optima_counts=[5]*10,
    ...                                                                 local_optima_counts=[5]*10)
    ...                         ))


    Here is how we'd tune the there free parameters to maximize the AUC measure:

    >>> from metrics import area_under_curve
    >>> result = tune_random(my_algorithm(pop_size=10), area_under_curve, 5,
    ...                      mutation_prob=(0.001, 0.2),
    ...                      mutation_std=(0.001, 0.1))
    >>> result # doctest:+ELLIPSIS
    <generator object ...>

    >>> print(*list(result), sep='\\n') # doctest:+ELLIPSIS
    (..., {'mutation_prob': ..., 'mutation_std': ...})
    (..., {'mutation_prob': ..., 'mutation_std': ...})
    (..., {'mutation_prob': ..., 'mutation_std': ...})
    (..., {'mutation_prob': ..., 'mutation_std': ...})
    (..., {'mutation_prob': ..., 'mutation_std': ...})

    """
    # TODO: Rewrite to use Dask
    # Take a Dask 'Client' object as an optional input, and use in conjunction with dask_jobqueue to support clusters:
    # https://jobqueue.dask.org/en/latest/
    best_config = None
    best = float('-inf')

    def run(**kwargs):
        result = list(algorithm(**kwargs))
        x, y = zip(*result)
        return metric(x, [ind.fitness for ind in y])

    for i in range(samples):
        configuration = {k: np.random.uniform(*v) for k, v in kwargs.items()}
        scores = [run(**configuration) for _ in range(tests_per_sample)]
        score = np.mean(scores)
        if score > best:
            best_config = configuration
            best = score
        yield (best, best_config)
