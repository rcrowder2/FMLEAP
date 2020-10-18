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

s
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

