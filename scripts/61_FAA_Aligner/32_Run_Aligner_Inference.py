#!/usr/bin/python3
import setup_run_dir    # this import tricks script to run from 2 levels up
import os
from   amrlib.alignments.faa_aligner import FAA_Aligner


if __name__ == '__main__':
    # Specify bin directory.  If bin_dir is specified it will use this directory, otherwise
    # it w will it will look for the environment variable FABIN_DIR or use the path.
    # bin_dir     = 'xxx'   # change commented line below
    working_dir = 'amrlib/data/working_faa_aligner/'
    #model_dir   = 'amrlib/data/model_aligner_faa'
    eng_fn      = os.path.join(working_dir, 'sents.txt')
    amr_fn      = os.path.join(working_dir, 'gstrings.txt')
    astrings_fn = os.path.join(working_dir, 'amr_alignment_strings.txt')
    surface_fn  = os.path.join(working_dir, 'amr_surface_aligned.txt')

    # Read in the english sentences and linearlized AMR lines
    print('Reading and writing data in', working_dir)
    with open(eng_fn) as f:
        eng_lines = [l.strip().lower() for l in f]
    with open(amr_fn) as f:
        amr_lines = [l.strip().lower() for l in f]

    # Decleare the inference class and run it
    #inference = FAA_Aligner(model_dir=model_dir, bin_dir=bin_dir)
    inference = FAA_Aligner()
    amr_surface_aligns, alignment_strings = inference.align_sents(eng_lines, amr_lines)

    # Save the final data
    print('Writing surface aligned amrs to', surface_fn)
    with open(surface_fn, 'w') as f:
        for line in amr_surface_aligns:
            f.write('%s\n' % line)

    print('Writing alignments strings to', astrings_fn)
    with open(astrings_fn, 'w') as f:
        for line in alignment_strings:
            f.write('%s\n' % line)
