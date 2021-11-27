#!/usr/bin/python3
import setup_run_dir    # Set the working directory and python sys.path to 2 levels above
import os
import penman
from   amrlib.graph_processing.amr_loading_raw import load_raw_amr
from   amrlib.alignments.penman_utils import test_for_decode_encode_issue, strip_surface_alignments


# Get rid of un-needed metadata and rename "alignments", "isi_alignments"
def mod_graph_meta(graph):
    id     = graph.metadata['id']
    tok    = graph.metadata['tok']
    aligns = graph.metadata['alignments']
    graph.metadata = {'id':id, 'tok':tok, 'isi_alignments':aligns}
    return graph


# Build a corpus of test cases for alignments
if __name__ == '__main__':
    corp_dir    = 'amrlib/data/amr_annotation_3.0/data/alignments/split/test'
    graph_fn    = 'amrlib/data/alignments/test_w_surface.txt'
    graph_ns_fn = 'amrlib/data/alignments/test_no_surface.txt'

    os.makedirs(os.path.dirname(graph_fn), exist_ok=True)

    # Loop through the files and load all entries
    entries = []
    print('Loading data from', corp_dir)
    fpaths = [os.path.join(corp_dir, fn) for fn in os.listdir(corp_dir)]
    for fpath in fpaths:
        entries += load_raw_amr(fpath)
    print('Loaded {:,} entries'.format(len(entries)))

    # Check for the penman decode/re-encode issue and strip some metadata
    good_graphs    = []
    good_graphs_ns = []
    for entry in entries:
        # Create a version with No Surface alignments
        entry_ns = strip_surface_alignments(entry)
        graph,    is_good    = test_for_decode_encode_issue(entry)
        graph_ns, is_good_ns = test_for_decode_encode_issue(entry_ns)
        if is_good and is_good_ns:
            good_graphs.append( mod_graph_meta(graph) )
            good_graphs_ns.append( mod_graph_meta(graph_ns) )

    # Save the collated data
    print('Saving {:,} good graphs to {:} and {:}'.format(len(good_graphs), graph_fn, graph_ns_fn))
    penman.dump(good_graphs,    graph_fn,    indent=6)
    penman.dump(good_graphs_ns, graph_ns_fn, indent=6)
