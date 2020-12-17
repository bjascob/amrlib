import os
import tempfile
import penman
from   penman.models.noop import NoOpModel
from   graphviz import Digraph    # sudo apt install graphviz; pip3 install graphviz


# render_fn is the temp file used for rendering
# format is pdf, png, ... (see graphviz supported formats)
class AMRPlot(object):
    def __init__(self, render_fn=None, format='pdf'):
        if render_fn is None:
            render_fn = os.path.join(tempfile.gettempdir(), 'amr_graph.gv')
        self.graph = Digraph('amr_graph', filename=render_fn, format=format)
        self.graph.attr(rankdir='LR', size='12,8') # rankdir=left-to-right, size=width,height in inches
        self.counter = 0

    # Build the AMR graph from a text entry
    # debug prints tuples to the scrieen
    # allow-deinvert enables the default penman decode behavior of de-inverting
    # edges ending in -of, such as 'arg0-of' to 'arg0'
    def build_from_graph(self, entry, debug=False, allow_deinvert=False):
        # Parse the AMR text
        if allow_deinvert:
            penman_graph = penman.decode(entry)
        else:
            model = NoOpModel()  # does not de-invert edges
            penman_graph = penman.decode(entry, model=model)
        # Build g.instances() => concept relations  (these are nodes)
        for t in penman_graph.instances():
            self._add_instance(t)
            if debug: print(t)
        # Build g.edges() => relations between nodes
        for t in penman_graph.edges():
            self._add_edge(t)
            if debug: print(t)
        # Build g.attributes  => relations between nodes and a constant
        for t in penman_graph.attributes():
            self._add_attribute(t)
            if debug: print(t)

    # Render graph to a file using the default filename in the initializer
    def render(self):
        png_fname = self.graph.render()
        return png_fname

    # Render the graph to the deafult filename and display it in the default viewer
    def view(self):
        self.graph.view()

    # Instances are nodes (circles with info) ie.. concept relations
    def _add_instance(self, t):
        # graph.node(name, label=None, _attributes=None, **attrs)
        label = str(t.source) + '/' + str(t.target)
        self.graph.node(t.source, label=label, shape='circle')

    # Edges are lines connecting nodes
    def _add_edge(self, t):
        # gaph.edge(tail_name, head_name, label=None, _attributes=None, **attrs)
        self.graph.edge(t.source, t.target,  label=t.role)

    # Attributes are relations (edge) connecting to a constant
    def _add_attribute(self, t):
        node_id = self.get_uid()
        self.graph.node(node_id, label=t.target, shape='rectangle')
        self.graph.edge(t.source, node_id,  label=t.role)

    # Get a unique ID for naming nodes
    def get_uid(self):
        uid = 'node_%d' % self.counter
        self.counter += 1
        return uid
