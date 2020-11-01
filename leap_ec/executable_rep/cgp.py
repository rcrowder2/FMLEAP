"""Cartesian genetic programming (CGP) representation."""
import networkx as nx

from leap_ec.decoder import Decoder
from .executable import Executable


##############################
# Class CGPExecutable
##############################
class CGPExecutable(Executable):
    """Represented a decoded CGP circuit, which can be executed on inputs."""

    def __init__(self, primitives, num_inputs, num_outputs, graph):
        assert(primitives is not None)
        assert(len(primitives) > 0)
        assert(num_inputs > 0)
        assert(num_outputs > 0)
        assert(graph is not None)
        self.primitives = primitives
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.graph = graph

    def __call__(self, *args):
        assert(len(args) == self.num_inputs)

        def get_sorted_input_nodes(node_id):
            """Pull all the nodes that feed into a given node, ordered by the
            port they feed into on the target node."""
            in_edges = list(self.graph.in_edges(node_id, data='order'))
            # Sort inputs by which "port" they are supposed to feed into
            in_edges = sorted(in_edges, key=lambda e: e[2])  # Element 2 is the 'order' attribute
            in_nodes = [ self.graph.nodes[e[0]] for e in in_edges ]
            return in_nodes


        # Assign the input values
        for i in range(self.num_inputs):
            self.graph.nodes[i]['output'] = args[i]
        
        # Compute the values of hidden nodes, going in order so preprequisites are computed first
        num_hidden = len(self.graph.nodes) - self.num_inputs - self.num_outputs
        for i in range(self.num_inputs, self.num_inputs + num_hidden):
            f = self.graph.nodes[i]['function']
            in_nodes = get_sorted_input_nodes(i)
            node_inputs = [ n['output'] for n in in_nodes ]
            self.graph.nodes[i]['output'] = f(*node_inputs)

        # Copy the output values into their nodes
        result = []
        for i in range(self.num_inputs + num_hidden, self.num_inputs + num_hidden + self.num_outputs):
            in_edges = list(self.graph.in_edges(i))
            assert(len(in_edges) == 1), f"CGP output node {i} is connected to {len(in_edges)} nodes, but must be connected to exactly 1."
            in_node = self.graph.nodes[in_edges[0][0]]
            oi = in_node['output']
            self.graph.nodes[i]['output'] = oi
            result.append(oi)

        return result


##############################
# Class CGPDecoder
##############################
class CGPDecoder(Decoder):
    """Implements the genotype-phenotype decoding for Cartesian genetic programming (CGP).
    
    A CGP genome is linear, but made up of one sub-sequence for each circuit elements.  In our
    version here, the first gene in each sub-sequence indicates the primitive (i.e., function) that node computes, 
    and the subsequence genes indicate the inputs to that primitive.
    
    For example, the sequence `[ 0, 2, 3 ]` indicates an element that computes the 0th primitive
    (as an index of the `primitives` list) and takes its inputs from nodes 2 and 3, respectively.
    
    """

    def __init__(self, primitives, num_inputs, num_outputs, num_layers, nodes_per_layer, max_arity, levels_back=None):
        assert(primitives is not None)
        assert(len(primitives) > 0)
        assert(num_inputs > 0)
        assert(num_outputs > 0)
        assert(num_layers > 0)
        assert(nodes_per_layer > 0)
        assert(max_arity > 0)
        self.primitives = primitives
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.nodes_per_layer = nodes_per_layer
        self.max_arity = max_arity
        self.levels_back = levels_back

    def num_genes(self):
        """The number of genes we expect to find in each genome."""
        return self.num_layers*self.nodes_per_layer*(self.max_arity + 1) + self.num_outputs

    def num_cgp_nodes(self):
        """Return the total number of nodes that will be in the resulting CGP graph, including inputs and outputs."""
        return self.num_inputs + self.num_layers*self.nodes_per_layer + self.num_outputs

    def get_primitive(self, genome, layer, node):
        """Given a linear CGP genome, return the primitive object for the given node in the 
        given layer."""
        assert(genome is not None)
        assert(layer >= 0)
        assert(layer < self.num_layers)
        assert(node >= 0)
        assert(node < self.nodes_per_layer)
        primitive_id = genome[(layer*self.nodes_per_layer + node)*(self.max_arity + 1)]
        assert(primitive_id < len(self.primitives)), f"The gene for node {node} of layer {layer} specifies a primitive function id {primitive_id}, but that's out of range: we only have {len(self.primitives)} primitive(s)!"
        return self.primitives[primitive_id]
    

    def get_input_sources(self, genome, layer, node):
        """Given a linear CGP genome, return the list of all of the input sources (as integers) 
        which feed into the given node in the given layer."""
        assert(genome is not None)
        assert(layer >= 0)
        assert(layer < self.num_layers)
        assert(node >= 0)
        assert(node < self.nodes_per_layer)


        def ith_input_gene(genome, layer, node, i):
            """Helper function that tells us which gene defines the ith input of
            the element at a given layer and node in a Cartesian circuit."""
            return (layer*self.nodes_per_layer + node)*(self.max_arity + 1) + 1 + i
        
        input_sources = []
        for i in range(self.max_arity):
            gene_id = ith_input_gene(genome, layer, node, i)
            input_sources.append(genome[gene_id])

        assert(len(input_sources) == self.max_arity)
        return input_sources

    def get_output_sources(self, genome):
        """Given a linear CGP genome, return the list of nodes that connect to each output."""
        first_output = self.num_layers*self.nodes_per_layer*(self.max_arity + 1)
        output_sources = genome[first_output:]
        return output_sources


    def decode(self, genome):
        """Decode a linear CGP genome into an executable circuit."""
        assert(genome is not None)
        assert(len(genome) == self.num_genes()), f"Expected a genome of length {self.num_genes()}, but was given one of length {len(genome)}."
        all_node_ids = [i for i in range(self.num_cgp_nodes())]

        graph = nx.DiGraph()
        graph.add_nodes_from(all_node_ids)

        # Add edges connecting interior nodes to their sources
        for layer in range(self.num_layers):
            for node in range(self.nodes_per_layer):
                # TODO Consider using Miller's pre-processing algorithm here to omit nodes that are disconnected from the circuit (making execution more efficient)
                node_id = self.num_inputs + layer*self.nodes_per_layer + node
                graph.nodes[node_id]['function'] = self.get_primitive(genome, layer, node)
                inputs = self.get_input_sources(genome, layer, node)
                # Mark each edge with an 'order' attribute so we know which port they feed into on the target node
                graph.add_edges_from([(i, node_id, {'order': o}) for o, i in enumerate(inputs)])

        # Add edges connecting outputs to their sources
        output_sources = self.get_output_sources(genome)
        output_nodes = all_node_ids[-self.num_outputs:]
        graph.add_edges_from(zip(output_sources, output_nodes))

        return CGPExecutable(self.primitives, self.num_inputs, self.num_outputs, graph)
                
