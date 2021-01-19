#!/usr/bin/python3
import setup_run_dir    # this import tricks script to run from 2 levels up
import os
from   amrlib.alignments.faa_aligner.postprocess import postprocess
from   amrlib.alignments.faa_aligner.proc_data import ProcData


if __name__ == '__main__':
    working_dir = 'amrlib/data/train_faa_aligner'
    astrings_fn = 'amr_alignment_strings.txt'
    surface_fn  = 'amr_surface_aligned.txt'

    print('Reading and writing data in', working_dir)

    # Load the original, preprocess and model output data
    data = ProcData.from_directory(working_dir)

    # Post process
    amr_surface_aligns, alignment_strings = postprocess(data)

    # Save the final data
    fpath = os.path.join(working_dir, astrings_fn)
    print('Writing alignments strings to', fpath)
    with open(fpath, 'w') as f:
        for line in alignment_strings:
            f.write('%s\n' % line)

    fpath = os.path.join(working_dir, surface_fn)
    print('Writing surface aligned amrs to', fpath)
    with open(fpath, 'w') as f:
        for line in amr_surface_aligns:
            f.write('%s\n' % line)
