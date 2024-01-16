#!/usr/bin/python3
import setup_run_dir    # this import tricks script to run from 2 levels up
import sys
from   amrlib.evaluate.smatch_enhanced import get_entries, compute_smatch, compute_scores
from   amrlib.evaluate.smatch_enhanced import redirect_smatch_errors

# Score "nowiki" version, meaning the generated file should not have the :wiki tags added
GOLD='amrlib/data/tdata_xfm/test.txt.nowiki'
PRED='amrlib/data/model_parse_xfm_bart_large/test-pred.txt'
# Score with the original version meaning the generated files need to have been "wikified"
#GOLD='amrlib/data/tdata_xfm/test.txt'
#PRED='amrlib/data/model_parse_xfm_bart_base/test-pred.txt.wiki'


redirect_smatch_errors('logs/score_smatch_errors.log')
# Run only the smatch score
if 0:
    gold_entries = get_entries(GOLD)
    test_entries = get_entries(PRED)
    precision, recall, f_score = compute_smatch(test_entries, gold_entries)
    print('SMATCH -> P: %.3f,  R: %.3f,  F: %.3f' % (precision, recall, f_score))
# Compute enhanced scoring
else:
    compute_scores(PRED, GOLD)
