#!/usr/bin/python3
import setup_run_dir    # this import tricks script to run from 2 levels up
import sys
# sys.stderr = open('logs/score_stderr.log', 'w')     # SMATCH AMR logs to stderr so redirect this to a file
# print('!!stderr redirected to logs/score_stderr.log.  See that file for graphs that did not load (and thus are not scored).')
import os
import logging
import penman
from   amrlib.graph_processing.amr_loading import load_amr_entries, get_graph_only
from   amrlib.utils.logging import setup_logging, WARN, ERROR
from   amrlib.evaluate.smatch_enhanced import get_entries, compute_smatch
from   amrlib.models.parse_t5.penman_serializer import PenmanDeSerializer

logger = logging.getLogger(__name__)


# Load a file where each line is an entry
# The first character indicates if the graph was clipped.
# return a set of indexes that where this is True
def load_lines(fname):
    with open(fname) as f:
        lines = f.readlines()
    clips    = [bool(int(l[0])) for l in lines]
    clips    = set([i for i, c in enumerate(clips) if c is True])
    gstrings = [l[2:].strip() for l in lines]
    return clips, gstrings


# This code is for debug only
# The code does the deserialization of graphs generated without deserialization
#
# Note tdata_gsii was created with 30_Model_Parse_GSII/10_Annotate_Corpus.py and 12_RemoveWikiData.py
# This can be changed.  The corpus doesn't need to be annotated (you can skip running 10_x) but
# wikidata should be removed since the model doesn't produce those tags and these graphs will be
# copied as the reference data to be scored at the end.
if __name__ == '__main__':
    setup_logging(logfname='logs/post_process.log', level=WARN)
    corpus_dir = 'amrlib/data/tdata_gsii/'
    ref_in_fn  = 'test.txt.features.nowiki'
    test_dir   = 'amrlib/data/test_parse_t5'
    gen_in_fn  = 'test.txt.generated'
    ref_out_fn = 'test.txt.reference.post'
    gen_out_fn = gen_in_fn + '.post'

    # Load the reference graphs
    fname = os.path.join(corpus_dir, ref_in_fn)
    print('Loading', fname)
    ref_amr_entries = load_amr_entries(fname)
    ref_in_graphs   = [get_graph_only(e) for e in ref_amr_entries]
    print('Loaded %d reference graphs' % len(ref_in_graphs))

    # Load the generated graphs
    fname = os.path.join(test_dir, gen_in_fn)
    print('Loading', fname)
    clip_index_set, gen_in_graphs = load_lines(fname)
    print('Loaded %d total generated graphs (%d clipped)' % (len(gen_in_graphs), len(clip_index_set)))

    # Check that the numbers line-up, or for test, truncate
    if 1:
        assert len(gen_in_graphs) == len(ref_in_graphs)
    else:   # for partial generate during debugging
        print('NOTE: Clipping reference set to match generated length - DEBUGGING ONLY')
        ref_in_graphs = ref_in_graphs[:len(gen_in_graphs)]

    # process the generated graphs through the deserializer
    gen_out_graphs = []
    bad_graphs     = set()
    for i, graph in enumerate(gen_in_graphs):
        # skip anything on the clips
        if i in clip_index_set:
            continue
        pen_graph = PenmanDeSerializer(graph, i).get_pen_graph()
        # If a graph doesn't deserialize, skip it
        if pen_graph is None:
            bad_graphs.add(i)
            logger.error('Decode-1 Err (graph %3d): %s' % (i, graph))
            continue
        # Add a test index so we can identify the graph
        pen_graph.metadata = {'test_id': str(i)}
        # Add the graph to good graphs - all exceptions should "continue" and not get here
        gen_out_graphs.append(pen_graph)
    num_non_clipped = len(gen_in_graphs) - len(clip_index_set)
    pct = 100.*len(bad_graphs)/num_non_clipped
    print('%d generated graphs do not deserialize out of %d = %.1f%%' % (len(bad_graphs), num_non_clipped, pct))
    print()

    # Save the reference, omitting any clipped or bad
    ref_fpath = os.path.join(test_dir, ref_out_fn)
    print('Saving', ref_fpath)
    skipped = 0
    with open(ref_fpath, 'w') as f:
        for i, graph in enumerate(ref_in_graphs):
            if i in bad_graphs or i in clip_index_set:
                skipped += 1
                continue
            # Add a test index so we can identify the graph
            f.write('# ::test_id %d\n' % i)
            f.write(graph + '\n\n')
    print('Skipped writing %d as either bad or clipped' % skipped)
    print('Wrote a total of %d reference AMR graphs' % (len(ref_in_graphs) - skipped))
    print()

    # Save the generated
    gen_fpath = os.path.join(test_dir, gen_out_fn)
    print('Saving', gen_fpath)
    penman.dump(gen_out_graphs, gen_fpath, indent=6)
    print('Wrote a total of %d generated AMR graphs' % len(gen_out_graphs))

    # Print some info
    print()
    print('Clipped: ', sorted(clip_index_set))
    print('Bad graphs: ', sorted(bad_graphs))
    print()

    # Score the resultant files
    print('Scoring the above files with SMATCH')
    gold_entries = get_entries(ref_fpath)
    test_entries = get_entries(gen_fpath)
    precision, recall, f_score = compute_smatch(test_entries, gold_entries)
    print('SMATCH -> P: %.3f,  R: %.3f,  F: %.3f' % (precision, recall, f_score))
