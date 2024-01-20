#!/usr/bin/env python3
import setup_run_dir    # this import tricks script to run from 2 levels up
import os
from   amrlib.graph_processing.annotator import annotate_file, load_spacy
from   amrlib.utils.logging import silence_penman


# Annotate the raw AMR entries with SpaCy to add the required ::tokens and ::lemmas fields
# plus a few other fields for future pre/postprocessing work that may be needed.
if __name__ == '__main__':
    silence_penman()
    indir  = 'amrlib/data/LDC2020T02'
    outdir = 'amrlib/data/tdata_generate_xfm'

    # Create the processed corpus directory
    os.makedirs(outdir, exist_ok=True)

    # Load the spacy model with the desired model
    load_spacy('en_core_web_sm')

    # run the pipeline
    for fn in ('test.txt', 'dev.txt', 'train.txt'):
        annotate_file(indir, fn, outdir, fn + '.features')
