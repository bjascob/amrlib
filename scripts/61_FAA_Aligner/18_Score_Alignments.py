#!/usr/bin/python3
import setup_run_dir    # this import tricks script to run from 2 levels up
from   amrlib.evaluate.alignment_scorer import AlignmentScorer, load_gold_alignments


# Simple file format where each lines is a space separated list of alignments
def load_oneline_alignments(fn, max_lines=200):
    alignments = []
    with open(fn) as f:
        for line in f:
            parts = [p.strip() for p in line.split()]
            parts = [p for p in parts if p]
            alignments.append(set(parts))
            if len(alignments) >= max_lines:
                break
    return alignments


# Note that the Hand alignments are for the LDC1 concensus files
# There are 100 entries in each test and dev for the LDC1 data and the Hand Alignments
if __name__ == '__main__':
    gold_dev_fn  = 'amrlib/alignments/isi_hand_alignments/dev-gold.txt'
    gold_test_fn = 'amrlib/alignments/isi_hand_alignments/test-gold.txt'
    aligned_fn   = 'amrlib/data/train_faa_aligner/amr_alignment_strings.txt'

    # Load the gold and model alignments
    # Dev are the first 100, the next 100 are test
    gold_dev_aligns,  _ = load_gold_alignments(gold_dev_fn)
    gold_test_aligns, _ = load_gold_alignments(gold_test_fn)
    model_alignments  = load_oneline_alignments(aligned_fn, 200)
    model_dev_aligns  = model_alignments[   :100]
    model_test_aligns = model_alignments[100:200]



    print('Scoring alignments from', aligned_fn)
    scores_dev  = AlignmentScorer(gold_dev_aligns,  model_dev_aligns)
    scores_test = AlignmentScorer(gold_test_aligns, model_test_aligns)
    print('Dev scores   ', scores_dev)
    print('Test scores  ', scores_test)
