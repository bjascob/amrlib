#!/usr/bin/python3
import setup_run_dir    # Set the working directory and python sys.path to 2 levels above
import re
import penman
from   penman.models.noop import NoOpModel
from   amrlib.graph_processing.amr_loading import load_amr_entries
from   amrlib.alignments.rbw_aligner import RBWAligner
from   amrlib.alignments.penman_utils import test_for_decode_encode_issue


# 11/27/2021: This test is currently broken


# Manual test to see if amrlib can generate the alignment string from surface alignments correctly,
# using the LDC data as the baseline
# !! Note that you must first create the test corpus.  See the scripts directory/Build_Aligment_Test_Corpus.py
if __name__ == '__main__':
    fname   = 'amrlib/data/alignments/test_w_surface.txt'
    entries = load_amr_entries(fname)

    # Loop through all entries and see if creating alignment string works
    if 1:
        # Filter out an entries that penman messes up
        # Looks like this is a simple de-inversion of some re-entrant nodes but that causes the addresses to change
        # This is not an issue of graphs already saved via penman, it's only an issue here because we want to look at the
        # alignments for the original graph to the ones that are re-created (re-created on a modified graph creates an issue).
        print('Testing for %d entries for ones that do not re-encode' % len(entries))
        bad_indexes = []
        for i, entry in enumerate(entries):
            matched = test_for_decode_encode_issue(entry)
            if not matched:
                bad_indexes.append( i )
        print('These indexes will be skipped: ', bad_indexes)
        print()

        # Test to see if old and new alignments match
        print('Testing match of new and original alignments')
        print('%d entries do not re-encode properly and will be skipped (counted as good).' % len(bad_indexes))
        num_bad     = 0
        bad_indexes = set(bad_indexes)
        for i, entry in enumerate(entries):
            if i in bad_indexes:
                continue
            graph   = penman.decode(entry, model=NoOpModel())
            graph.metadata['old_aligns'] = graph.metadata['isi_alignments']
            del graph.metadata['isi_alignments']
            RBWAligner.add_alignment_string(graph, 'new_aligns')
            # compare as a set because when there are nodes with x.y and x.y.r the ordering may be different
            old_aligns = set(graph.metadata['old_aligns'].split())
            new_aligns = set(graph.metadata['new_aligns'].split())
            if  old_aligns != new_aligns:
                num_bad += 1
                print('index', i)
                print('# ::id', graph.metadata['id'])
                print('# ::old_aligns', graph.metadata['old_aligns'])
                print('# ::new_aligns', graph.metadata['new_aligns'])
                print()
        print('There are %d bad alignments out of %d total' % (num_bad, len(entries)))

    # Print a specific entry for debug
    else:
        index   = 0 #1095
        entry   = entries[index]
        graph   = penman.decode(entry, model=NoOpModel())
        graph.metadata['old_aligns'] = graph.metadata['alignments']
        del graph.metadata['alignments']

        add_alignment_string(graph)
        print(penman.encode(graph, indent=6))
        print()
