#!/usr/bin/python3
import setup_run_dir    # Set the working directory and python sys.path to 2 levels above
import re
import logging
import json
from   multiprocessing import Pool
from   functools import partial
import spacy
import penman
from   penman.models.noop import NoOpModel
from   amrlib.utils.logging import setup_logging, silence_penman, WARN
from   amrlib.graph_processing.amr_loading import load_amr_entries
from   amrlib.graph_processing.annotator import load_spacy, add_lemmas
from   amrlib.alignments.rbw_aligner import RBWAligner

logger = logging.getLogger(__name__)


# Run the aligner on the LDC files with existing alignments for comparison
# The ISI hand alignments are for LDC2014T12 (AMR1) the test-concensus.txt and dev-concensus.txt files
if __name__ == '__main__':
    setup_logging(level=WARN, logfname='logs/rbw_aligner.log')
    silence_penman()

    in_fname  = 'amrlib/data/amr_annotation_1.0/data/split/test/amr-release-1.0-test-consensus.txt'
    out_fname = 'amrlib/data/alignments/test-aligned.txt'
    # in_fname  = 'amrlib/data/amr_annotation_1.0/data/split/dev/amr-release-1.0-dev-consensus.txt'
    # out_fname = 'amrlib/data/alignments/dev-aligned.txt'

    # Load and convert to a penman graph
    print('Loading', in_fname)
    entries = load_amr_entries(in_fname)
    print('Loaded %d entries' % len(entries))

    # Convert to penman and add lemmas
    print('Annotating')
    load_spacy()    # do this in the main process to prevent doing it multiple times
    graphs = []
    annotate = partial(add_lemmas, snt_key='snt', verify_tok_key=None)      # no existing tok key
    with Pool() as pool:
        for graph in pool.imap(annotate, entries):
            if graph is not None:
                graphs.append(graph)
    print('%d graphs left with the same tokenization length' % len(graphs))

    # Run the aligner
    print('Aligning Graphs')
    new_graphs = []
    keep_keys = ('id', 'snt', 'tokens', 'lemmas', 'rbw_alignments')
    for graph in graphs:
        aligner = RBWAligner.from_penman_w_json(graph, align_str_name='rbw_alignments')
        pgraph = aligner.get_penman_graph()
        pgraph.metadata = {k:v for k, v in pgraph.metadata.items() if k in keep_keys}
        new_graphs.append( pgraph )

    # Save the graphs
    print('Saving to', out_fname)
    penman.dump(new_graphs, out_fname, model=NoOpModel(), indent=6)
