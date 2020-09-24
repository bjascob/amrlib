import re
import penman
from   penman.models.noop import NoOpModel


# Penman has a bug or two that prevents graphs string from being decoded and then re-encoded
# to the same string, even when using the NoOpModel.
# See https://github.com/goodmami/penman/issues/94
def test_for_decode_encode_issue(gold):
    graph = penman.decode(gold, model=NoOpModel())
    test  = penman.encode(graph, indent=6, compact=True, model=NoOpModel())
    gold  = to_graph_line(gold)
    test  = to_graph_line(test)
    is_good = test == gold
    return graph, is_good

# Utility function for the above
def to_graph_line(e):
    lines  = [l.strip() for l in e.splitlines()]
    lines  = [l for l in lines if (l and not l.startswith('#'))]
    string = ' '.join(lines)
    string = string.replace('\t', ' ')
    string = re.sub(' +', ' ', string)
    return string.strip()

# Strip the surface alignments from the graph string
def strip_surface_alignments(graph_string):
    # Match ~e. plus 1 or more integers plus zero or more of the patterns ,\d+
    # ie, strip  ~.e4,5,10
    return re.sub(r'~e\.\d+[,\d+]*', '', graph_string)
