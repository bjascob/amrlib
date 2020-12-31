#!/usr/bin/python3
import setup_run_dir    # this import tricks script to run from 2 levels up
import os
import json
from   nltk.tokenize import word_tokenize
from   amrlib.evaluate.bleu_scorer import BLEUScorer


# Read file with 1 sentence per line
def read_file(fname):
    print('Reading ', fname)
    sents = []
    with open(fname) as f:
        for line in f:
            line = line.strip().lower()
            if line:
                tokens = word_tokenize(line)
                sents.append(tokens)
    return sents


# Take 2 files with 1 sentence per line, and load them.
# Run the tokenized sentences through the BLEU scorer.
if __name__ == '__main__':
    model_dir = 'amrlib/data/model_generate_t5wtense'

    golden_file = os.path.join(model_dir, 'test.txt.ref_sents.tagged.clipped.beam16')
    pred_file   = os.path.join(model_dir, 'test.txt.generated.tagged.clipped.beam16')

    # Read in the two files
    refs  = read_file(golden_file)
    preds = read_file(pred_file)

    # Score (at this point preds and refs are lists of lists)
    assert len(preds) == len(refs), '%s != %s' % (len(preds), len(refs))
    bleu_scorer = BLEUScorer()
    bleu_score, _, _ = bleu_scorer.compute_bleu(refs, preds)
    print('BLEU score: %5.2f' % (bleu_score*100.))
