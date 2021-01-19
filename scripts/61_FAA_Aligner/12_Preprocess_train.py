#!/usr/bin/python3
import setup_run_dir    # this import tricks script to run from 2 levels up
import os
from   amrlib.alignments.faa_aligner.preprocess import preprocess_train


if __name__ == '__main__':
    working_dir = 'amrlib/data/train_faa_aligner'
    eng_fn      = os.path.join(working_dir, 'sents.txt')
    amr_fn      = os.path.join(working_dir, 'gstrings.txt')
    fa_in_fn    = os.path.join(working_dir, 'fa_in.txt')

    print('Reading and writing data in', working_dir)
    # Read in the english sentences and linearized AMR lines
    with open(eng_fn) as f:
        eng_lines = [l.strip().lower() for l in f]
    with open(amr_fn) as f:
        amr_lines = [l.strip().lower() for l in f]

    # Proprocess the data
    eng_td_lines, amr_td_lines = preprocess_train(working_dir, eng_lines, amr_lines)

    # Save in fast align training format
    with open(fa_in_fn, 'w') as f:
        for en_line, amr_line in zip(eng_td_lines, amr_td_lines):
            f.write('%s ||| %s\n' % (en_line, amr_line))
