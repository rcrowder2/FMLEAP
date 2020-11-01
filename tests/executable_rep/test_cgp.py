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


@pytest.fixture
def test_2layer_phenome():
    """A simple CGP circuit that computes AND and is made up of four NAND gates."""
    nand = lambda x, y: not (x and y)
    primitives = [ nand ]
    genome = [ 0, 0, 1,  # Node 2
               0, 1, 0,  # Node 3
               0, 2, 3,  # Node 4
               0, 3, 2,  # Node 5
               5 ]  # Output is node 5

    decoder = cgp.CGPDecoder(primitives=primitives,
                             num_inputs=2,
                             num_outputs=1,
                             num_layers=2,
                             nodes_per_layer=2,
                             max_arity=2)
    
    return decoder.decode(genome)


def test_decode2(test_2layer_phenome):
    """When primitives have arity > 1, the edges of the decoded graph should have an `order` 
    attribute that correctly indicates which input the feed to on the destination node."""
    assert(test_2layer_phenome.num_inputs == 2)
    assert(test_2layer_phenome.num_outputs == 1)

    graph = test_2layer_phenome.graph

    assert(7 == graph.number_of_nodes())
    assert(9 == graph.number_of_edges())
    assert(graph.has_edge(0, 2))
    assert(graph.has_edge(1, 2))
    assert(graph.has_edge(0, 3))
    assert(graph.has_edge(1, 3))
    assert(graph.has_edge(2, 4))
    assert(graph.has_edge(3, 4))
    assert(graph.has_edge(2, 5))
    assert(graph.has_edge(3, 5))

    # Each internal node should have arity of 2
    for i in [2, 3, 4, 5]:
        assert(2 == len(list(graph.in_edges(i))))

    # The output node takes only one input
    assert(1 == len(list(graph.in_edges(6))))

    # Check that the edges are feeding into the correct ports
    assert(graph.edges[0, 2]['order'] == 0)  # Input 0 feeds into the 0th port of node 2
    assert(graph.edges[1, 2]['order'] == 1)  # Input 1 feeds into the 1st port of node 2
    assert(graph.edges[0, 3]['order'] == 1)  # Input 0 feeds into the 1st port of node 3
    assert(graph.edges[1, 3]['order'] == 0)  # Input 1 feeds into the 0th port of node 3


##############################
# Tests for CGPExecutable
##############################
@pytest.fixture
def tt_inputs():
    """Returns the 4 permutations of Boolean inputs for a 2-input truth table."""
    return [ [ True, True ],
             [ True, False ],
             [ False, True ],
             [ False, False ] ]


def test_call1(test_2layer_phenome, tt_inputs):
    """The test individuals should compute the AND function."""
    assert(test_2layer_phenome.num_inputs == 2)
    assert(test_2layer_phenome.num_outputs == 1)

    # Truth table for AND
    expected = [ [True], [False], [False], [False] ]

    result = [ test_2layer_phenome(*in_vals) for in_vals in tt_inputs ]

    assert(expected == result)


