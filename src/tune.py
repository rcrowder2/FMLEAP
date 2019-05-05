import numpy as np
from dask import delayed
from dask.distributed import Client


def tune_random(algorithm, metric, samples, client, tests_per_sample=5, **kwargs):
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

    Now we require a :py:mod:`dask` :py:class:`Client` to tell the parameter tuner how to parallelize its runs.  We'll
    stick with the default client here, which creates separate processes on the local machine, one per core:

    >>> from dask.distributed import Client

    Here is how we'd tune the there free parameters to maximize the AUC measure:

    >>> from metrics import area_under_curve
    >>> scores, configurations = tune_random(my_algorithm(pop_size=10), area_under_curve, 5, Client(),
    ...                                      mutation_prob=(0.001, 0.2),
    ...                                      mutation_std=(0.001, 0.1))

    >>> print(*zip(scores, configurations), sep='\\n') # doctest:+ELLIPSIS
    (..., {'mutation_prob': ..., 'mutation_std': ...})
    (..., {'mutation_prob': ..., 'mutation_std': ...})
    (..., {'mutation_prob': ..., 'mutation_std': ...})
    (..., {'mutation_prob': ..., 'mutation_std': ...})
    (..., {'mutation_prob': ..., 'mutation_std': ...})

    """

    @delayed
    def run(**kwargs):
        result = list(algorithm(**kwargs))
        x, y = zip(*result)
        return metric(x, [ind.fitness for ind in y])

    scores = []
    configurations = []
    for i in range(samples):
        configuration = {k: np.random.uniform(*v) for k, v in kwargs.items()}
        config_scores = [run(**configuration) for _ in range(tests_per_sample)]

        score = delayed(np.mean)(config_scores)
        scores.append(score)
        configurations.append(configuration)
        scores = client.gather(client.compute(scores))
    return scores, configurations

