#!/usr/bin/env python3
import setup_run_dir    # this import tricks script to run from 2 levels up
import os
import json
from   nltk.tokenize import word_tokenize
from   amrlib.evaluate.bleu_scorer import BLEUScorer


# Read file with 1 sentence per line
def read_file(fname):
    print('Reading ', fname)
    with open(fname) as f:
        sents = [s.strip() for s in f]
    sents = [s for s in sents if s]
    return sents


def score_bleu(preds, refs):
    assert len(preds) == len(refs)
    # Lower-case and word_tokenize
    refs  = [word_tokenize(s.strip().lower()) for s in refs]
    preds = [word_tokenize(s.strip().lower()) for s in preds]
    bleu_scorer = BLEUScorer()
    bleu_score, _, _ = bleu_scorer.compute_bleu(refs, preds)
    return bleu_score


# Score 2 files with the same number of sentences in them
if __name__ == '__main__':
    model_dir = 'amrlib/data/model_generate_xfm'
    ref_file  = os.path.join(model_dir, 'test.txt.reference_sents.tagged.clipped.beam1')
    pred_file = os.path.join(model_dir, 'test.txt.generated_sents.tagged.clipped.beam1')

    # Read in the two files
    refs  = read_file(ref_file)
    preds = read_file(pred_file)

    # Score (at this point preds and refs are lists of lists)
    assert len(preds) == len(refs)
    bleu_score = score_bleu(preds, refs)
    print('BLEU score: %5.2f' % (bleu_score*100.))
