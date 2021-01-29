#!/usr/bin/python3
import setup_run_dir    # this import tricks script to run from 2 levels up
import os
from   amrlib.alignments.faa_aligner.preprocess import preprocess_train


if __name__ == '__main__':
    working_dir = 'amrlib/data/working_isi_aligner'
    eng_fn      = os.path.join(working_dir, 'sents.txt')
    amr_fn      = os.path.join(working_dir, 'gstrings.txt')

    print('Reading and writing data in', working_dir)
    # Read in the english sentences and linearized AMR lines
    with open(eng_fn) as f:
        eng_lines = [l.strip().lower() for l in f]
    with open(amr_fn) as f:
        amr_lines = [l.strip().lower() for l in f]

    # Proprocess the data
    data = preprocess_train(eng_lines, amr_lines)

    # Save the preprocess data and the model input file, the input data
    # already in the working directory
    eng_tok_pos_fn  = os.path.join(working_dir, 'eng_tok_origpos.txt')
    amr_tuple_fn    = os.path.join(working_dir, 'amr_tuple.txt')
    eng_model_in_fn = os.path.join(working_dir, 'en')
    amr_model_in_fn = os.path.join(working_dir, 'fr')
    data.save_lines(eng_tok_pos_fn,  data.eng_tok_origpos_lines)
    data.save_lines(amr_tuple_fn,    data.amr_tuple_lines)
    data.save_lines(eng_model_in_fn, data.eng_preproc_lines)
    data.save_lines(amr_model_in_fn, data.amr_preproc_lines)
