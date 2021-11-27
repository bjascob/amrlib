#!/usr/bin/python3
import setup_run_dir    # this import tricks script to run from 2 levels up
import sys
#sys.stderr = open('logs/score_stderr.log', 'w')     # SMATCH AMR logs to stderr so redirect this to a file
#print('!!stderr redirected to logs/score_stderr.log.  See that file for graphs that did not load (and thus are not scored).')
#print('  grep "Please check" logs/score_stderr.log | wc -l  to see how many this impacts.  Skipping could raise scores')
from   amrlib.evaluate.smatch_enhanced import get_entries, compute_smatch, compute_scores

# Score "nowiki" version, meaning the generated file should not have the :wiki tags added
GOLD='amrlib/data/tdata_t5/test.txt'
PRED='amrlib/data/model_parse_t5/test-pred.txt.wiki'

# Run only the smatch score
if 0:
    gold_entries = get_entries(GOLD)
    test_entries = get_entries(PRED)
    precision, recall, f_score = compute_smatch(test_entries, gold_entries)
    print('SMATCH -> P: %.3f,  R: %.3f,  F: %.3f' % (precision, recall, f_score))
# Compute enhanced scoring
else:
    compute_scores(GOLD, PRED)
