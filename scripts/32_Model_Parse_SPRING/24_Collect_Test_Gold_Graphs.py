#!/usr/bin/python3
import setup_run_dir    # Set the working directory and python sys.path to 2 levels above
import os
from   glob import glob
from   amrlib.graph_processing.amr_loading_raw import load_raw_amr


# Collect all the amr graphs from multiple files and create a gold test file.
# This simply concatenates files and cleans a few bad characters out.  The glob pattern
# needs to be exactly the same as what's in generate so the output graph ordering is the same.
if __name__ == '__main__':
    glob_pattern = 'amrlib/data/amr_annotation_3.0/data/amrs/split/test/*.txt'
    out_fpath    = 'amrlib/data/model_parse_spring/test-gold.txt.wiki'

    # Load the data
    graphs = []
    print('Loading data from', glob_pattern)
    for fpath in sorted(glob(glob_pattern)):
        graphs.extend(load_raw_amr(fpath))
    print('Loaded {:,} graphs'.format(len(graphs)))

    # Save the collated data
    print('Saving data to', out_fpath)
    with open(out_fpath, 'w') as f:
        for graph in graphs:
            f.write('%s\n\n' % graph)
    print()
