#!/usr/bin/python3
import setup_run_dir    # Set the working directory and python sys.path to 2 levels above
import os
from   glob import glob
from   amrlib.graph_processing.amr_loading_raw import load_raw_amr
from   amrlib.graph_processing.wiki_remover import wiki_remove_file
from   amrlib.utils.logging import silence_penman


# Load a list list of amr files as describe by the glob pattern and write
# them to a single file.
def load_and_resave_amr_files(in_glob_pattern, out_fpath):
    # Load the files
    graphs = []
    print('Loading data from', in_glob_pattern)
    for fpath in sorted(glob(in_glob_pattern)):
        graphs.extend(load_raw_amr(fpath))
    print('Loaded {:,} graphs'.format(len(graphs)))
    # Save the collated data
    print('Saving data to', out_fpath)
    with open(out_fpath, 'w') as f:
        for graph in graphs:
            f.write('%s\n\n' % graph)
    print()
    return graphs


# Collect all the amr graphs from multiple files and create a gold test file.
# This simply concatenates files and cleans a few bad characters out.  The glob pattern
# needs to be exactly the same as what's in generate so the output graph ordering is the same.
if __name__ == '__main__':
    silence_penman()
    base_amr_dir = 'amrlib/data/amr_annotation_3.0/data/amrs/split'
    out_dir      = 'amrlib/data/tdata_t5'

    # Create the output directory
    os.makedirs(out_dir, exist_ok=True)

    # Loop through the file tiles
    for amr_sub_dir in ('dev', 'test', 'training'):
        in_glob_pattern = os.path.join(base_amr_dir, amr_sub_dir, '*.txt')
        out_fn    = 'train.txt' if amr_sub_dir == 'training' else amr_sub_dir + '.txt'
        out_fpath = os.path.join(out_dir, out_fn)
        load_and_resave_amr_files(in_glob_pattern, out_fpath)
        print('Removing wiki')
        wiki_remove_file(out_dir, out_fn, out_dir, out_fn + '.nowiki')
        print()
