#!/usr/bin/env python3

import click
import toolz

import real
import operators as op


def evolve_mu_plus_lambda(evals, initializer, parent_selector, pipeline, evaluator):
    population = initializer()
    population = list(evaluator(population))

    i = 0
    while i < evals:
        # Selects parents
        parents = list(parent_selector(population))

        # Create a generator that creates offspring
        offspring = toolz.pipe(parents, *pipeline)
        offspring = list(offspring)  # Execute it

        population = parents + evaluator(offspring)
        i += len(offspring)

        yield (i, op.best(population))


@click.command()
@click.option('--mu', type=int, default=1, show_default=True, help='The parent population size for the (μ + λ) EA.')
@click.option('--lambda', 'lambda_', default=4, show_default=True, type=int, help='The offspring population size for the (μ + λ) EA.')
@click.option('--l', default=10, show_default=True, type=int, help='The genome length.')
@click.option('--mutation_prob', default=None, type=float, help='Gene-wise probability of mutation.  Defaults 1/genome_length.')
@click.option('--mutation_std', default=0.05, show_default=True, type=float, help='σ of the Gaussian mutation operator.')
@click.option('--evals', default=1000, show_default=True, type=int, help='The number of evaluations to run for.')
def run(mu, lambda_, l, mutation_prob, mutation_std, evals):
    """Execute a basic (μ + λ)-style EA on a real-valued fitness function."""
    assert(mu > 0)
    assert(lambda_ > 0)
    assert((not mutation_prob) or (mutation_prob >= 0))
    assert(mutation_std >= 0)
    assert(evals > 0)

    # We default to a 1/l mutation rate
    if not mutation_prob:
        mutation_prob = 1/l

    # Set up a function to produce the initial population
    initializer = real.random_initializer(mu + lambda_, l)

    # Set up a function to select μ parents
    parent_selector = op.truncation(mu=mu)

    # Set up the operator pipeline
    pipeline = [
        op.cloning(offspring_per_ind=int(lambda_/mu)),
        op.mutate_gaussian(prob=mutation_prob, std=mutation_std)
    ]

    # Set up the fitness function
    fun = real.cosine_family(alpha=0.6, global_optima_counts=[5]*l, local_optima_counts=[5]*l)
    evaluator = op.evaluate(fitness_function=fun)

    # Setup the EA
    result = evolve_mu_plus_lambda(evals,
                                   initializer=initializer,
                                   parent_selector=parent_selector,
                                   pipeline=pipeline,
                                   evaluator=evaluator)

    # Print the result
    print("Eval, Best_Fitness")
    for i, best in result:
        print(f"{i}, {best.fitness}")


if __name__ == '__main__':
    run()
