#!/usr/bin/python3
import setup_run_dir    # this import tricks script to run from 2 levels up
from   amrlib.evaluate.smatch_enhanced import compute_scores

# Score "nowiki" version, meaning the generated file should not have the :wiki tags added
GOLD='amrlib/data/tdata_gsii/test.txt.features.nowiki'
PRED='amrlib/data/model_parse_gsii/epoch200.pt.test_generated'

# Score with the test files with :wiki tags
#GOLD='amrlib/data/tdata_gsii/test.txt.features'
#PRED='amrlib/data/model_parse_gsii/epoch200.pt.test_generated.wiki'

compute_scores(PRED, GOLD)
