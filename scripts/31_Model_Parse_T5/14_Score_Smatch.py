#!/usr/bin/python3
import setup_run_dir    # this import tricks script to run from 2 levels up
import sys
# sys.stderr = open('logs/score_stderr.log', 'w')     # SMATCH AMR logs to stderr so redirect this to a file
# print('!!stderr redirected to logs/score_stderr.log.  See that file for graphs that did not load (and thus are not scored).')
from   amrlib.evaluate.smatch_enhanced import get_entries, compute_smatch

# Score "nowiki" version, meaning the generated file should not have the :wiki tags added
GOLD='amrlib/data/model_parse_t5/test.txt.reference'
PRED='amrlib/data/model_parse_t5/test.txt.generated'

# Run only the smatch score
gold_entries = get_entries(GOLD)
test_entries = get_entries(PRED)
precision, recall, f_score = compute_smatch(test_entries, gold_entries)
print('SMATCH -> P: %.3f,  R: %.3f,  F: %.3f' % (precision, recall, f_score))
