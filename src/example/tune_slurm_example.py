from toolz import curry
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

from evolve import generational
from metrics import area_under_curve
import operators as op
import real
import tune


@curry
def my_algorithm(pop_size, mutation_prob, mutation_std):
    return generational(evals=10000,
                        initialize=real.random_initializer(pop_size, genome_length=10),
                        pipeline=[
                             op.tournament(n=pop_size),
                             op.cloning,
                             op.mutate_gaussian(prob=mutation_prob, std=mutation_std)
                        ],
                        evaluate=op.evaluate(
                            fitness_function=real.cosine_family(alpha=0.6,
                                                                global_optima_counts=[5]*10,
                                                                local_optima_counts=[5]*10)
                        ))


if __name__ == '__main__':
    cluster = SLURMCluster(processes=2,
                           cores=2,
                           memory='500MB',
                           walltime='00:05:00',
                           local_directory='/scratch/escott8',
                           job_extra=['-o /scratch/escott8/test.out'])
    print('α')
    cluster.scale(1)
    print(cluster.job_script())
    print('β')
    client = Client(cluster)
    print('γ')
    scores, configurations = tune.tune_random(my_algorithm(pop_size=10), area_under_curve, 5, client,
                                              mutation_prob=(0.001, 0.2),
                                              mutation_std=(0.001, 0.1))
    print('δ')
    x = client.compute(scores)
    print('ε')
    y = client.gather(x)
    print('ζ')
    with open('out.txt', 'w') as f:
        [f.write(s) for s in x]

    cluster.close()
