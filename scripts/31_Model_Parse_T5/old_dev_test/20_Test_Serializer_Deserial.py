#!/usr/bin/python3
import setup_run_dir    # this import tricks script to run from 2 levels up
import sys
# sys.stderr = open('logs/score_stderr.log', 'w')     # SMATCH AMR logs to stderr so redirect this to a file
# print('!!stderr redirected to logs/score_stderr.log.  See that file for graphs that did not load (and thus are not scored).')
import os
import logging
import penman
from   penman.models.noop import NoOpModel
from   amrlib.graph_processing.amr_loading import load_amr_entries, get_graph_only
from   amrlib.utils.logging import setup_logging, WARN, DEBUG
from   amrlib.evaluate.smatch_enhanced import get_entries, compute_smatch
from   amrlib.models.parse_t5.penman_serializer import PenmanDeSerializer, load_and_serialize  

logger = logging.getLogger(__name__)


# Code to take the reference graphs, serialize them and then deserialize
# The only purpose of this code is to test the effectiveness of penman_serializer.py code
# Ideally the process would be lossless, giving a SMATCH score of 1.0
if __name__ == '__main__':
    setup_logging(logfname='logs/serial_deserial.log', level=WARN)
    corpus_dir = 'amrlib/data/LDC2020T02'
    in_fn      = 'test.txt'
    out_dir    = 'amrlib/data/test_parse_t5'
    ref_out_fn = in_fn + '.roundtrip_ref'
    gen_out_fn = in_fn + '.roundtrip_gen'

    # Make the out directory
    os.makedirs(out_dir, exist_ok=True)

    # Load the reference graphs
    fname = os.path.join(corpus_dir, in_fn)
    print('Loading', fname)
    ref_amr_entries = load_amr_entries(fname)
    ref_in_graphs   = [get_graph_only(e) for e in ref_amr_entries]
    print('Loaded %d reference graphs' % len(ref_in_graphs))

    # Simulate the generated graphs by running the original references throught the serializer
    print('Reloading and serializing', fname)
    gen_in_graphs  = load_and_serialize(fname)
    gen_in_graphs  = gen_in_graphs['serials']
    clip_index_set = set()  # empty - just so code below is a copy of post-process for generation

    # process the generated graphs through penman
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
            logger.warning('Decode-1 Err: %s' % graph)
            continue
        # Add the graph to good graphs - all exceptions should "continue" and not get here
        gen_out_graphs.append(pen_graph)
    num_non_clipped = len(gen_in_graphs) - len(clip_index_set)
    pct = 100.*len(bad_graphs)/num_non_clipped
    print('%d generated graphs do not deserialize out of %d = %.1f%%' % (len(bad_graphs), num_non_clipped, pct))
    print()

    # Save the reference, omitting any clipped or bad
    ref_fpath = os.path.join(out_dir, ref_out_fn)
    print('Saving', ref_fpath)
    skipped = 0
    with open(ref_fpath, 'w') as f:
        for i, graph in enumerate(ref_in_graphs):
            if i in bad_graphs or i in clip_index_set:
                skipped += 1
                continue
            f.write(graph + '\n\n')
    print('Skipped writing %d as either bad or clipped' % skipped)
    print('Wrote a total of %d reference AMR graphs' % (len(ref_in_graphs) - skipped))
    print()

    # Save the generated
    gen_fpath = os.path.join(out_dir, gen_out_fn)
    print('Saving', gen_fpath)
    penman.dump(gen_out_graphs, gen_fpath, indent=6, model=NoOpModel())
    print('Wrote a total of %d generated AMR graphs' % len(gen_out_graphs))
    print()

    # Score the resultant files
    print('Scoring the above files with SMATCH')
    gold_entries = get_entries(ref_fpath)
    test_entries = get_entries(gen_fpath)
    precision, recall, f_score = compute_smatch(test_entries, gold_entries)
    print('SMATCH -> P: %.3f,  R: %.3f,  F: %.3f' % (precision, recall, f_score))
