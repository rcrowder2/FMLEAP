"""Example demonstrating the use of Cartesin genetic programming (CGP) to 
evolve logic circuits to solve Boolean functions."""
import sys

from leap_ec.algorithm import generational_ea
from leap_ec import ops, probe, context
from leap_ec.representation import Representation
from leap_ec.executable_rep import cgp, problems

if __name__ == '__main__':
    pop_size = 5

    # The CGPDecoder is the heart of our CGP representation.
    # We'll set it up first because it's needed as a parameter 
    # to a few different components.
    cgp_decoder = cgp.CGPDecoder(
                        primitives=[
                            lambda x, y: not (x and y),  # NAND
                            lambda x, y: not x,  # NOT (ignoring y)
                        ],
                        num_inputs = 2,
                        num_outputs = 1,
                        num_layers=50,
                        nodes_per_layer=1,
                        max_arity=2
                    )

    with open('./cgp_stats.csv', 'w') as log_stream:
        ea = generational_ea(500, pop_size, 

                representation=Representation(
                    decoder=cgp_decoder,
                    # We use a sepecial initializer that obeys the CGP constraints
                    initialize=cgp.create_cgp_vector(cgp_decoder)
                ),

                # Our fitness function will be to solve the XOR problem
                problem=problems.TruthTableProblem(
                    boolean_function=lambda x: [ x[0] ^ x[1] ],  # XOR
                    num_inputs = 2,
                    num_outputs = 1
                ),

                pipeline=[
                    ops.tournament_selection,
                    ops.clone,
                    cgp.cgp_mutate(cgp_decoder),
                    ops.evaluate,
                    ops.pool(size=pop_size),
                    probe.FitnessStatsCSVProbe(context.context, stream=sys.stdout)
                ]
        )

        list(ea)