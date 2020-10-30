import pytest
import networkx as nx

from leap_ec.executable_rep import cgp


##############################
# Tests for CGPDecoder
##############################
def test_num_genes1():
    """A linear genome with just one input, node, and output, and arity of 1 should 
    have 3 genes."""
    decoder = cgp.CGPDecoder(primitives=[lambda x: not x], num_inputs=1, num_outputs=1, num_layers=1, nodes_per_layer=1, max_arity=1)
    assert(3 == decoder.num_genes())


def test_decode1():
    """A linear genome with just one input, node, and output, should yield a
    graph connected all three."""
    genome = [0, 0, 1]
    decoder = cgp.CGPDecoder(primitives=[lambda x: not x], num_inputs=1, num_outputs=1, num_layers=1, nodes_per_layer=1, max_arity=1)
    phenome = decoder.decode(genome)

    assert(3 == phenome.graph.number_of_nodes())
    assert(2 == phenome.graph.number_of_edges())
    assert(phenome.graph.has_edge(0, 1))
    assert(phenome.graph.has_edge(1, 2))

def test_decode2():
    """When primitives have arity > 1, the edges of the decoded graph should have a `input_order` 
    attribute that correctly indicates which input the feed to on the destination node."""
    assert(False), "TODO"


##############################
# Tests for CGPExecutable
##############################
@pytest.fixture
def test_ind():
    """A simple CGP circuit that computes AND and is made up of four NAND gates."""
    nand = lambda x, y: not (x and y)
    primitives = [ nand ]
    genome = [ 0, 0, 1,
               0, 0, 1,
               0, 2, 3,
               0, 2, 3 ]

    decoder = cgp.CGPDecoder(primitives=primitives,
                             num_inputs=2,
                             num_outputs=1,
                             num_layers=1,
                             nodes_per_layer=1,
                             max_arity=2)
    
    return decoder.decode(genome)


@pytest.fixture
def tt_inputs():
    """Returns the 4 permutations of Boolean inputs for a 2-input truth table."""
    return [ [ True, True ],
             [ True, False ],
             [ False, True ],
             [ False, False ] ]


def test_call1(test_ind, tt_inputs):
    """The test individuals should compute the AND function."""
    assert(test_ind.num_inputs == test_ind.num_inputs)
    assert(test_ind.num_outputs == test_ind.num_outputs)

    # Truth table for AND
    expected = [ True, False, False, False ]

    result = [ test_ind(*in_vals) for in_vals in tt_inputs ]

    assert(expected == result)


