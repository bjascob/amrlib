#!/usr/bin/python3
import setup_run_dir    # Set the working directory and python sys.path to 2 levels above
import os
from   glob import glob
from   amrlib.graph_processing.amr_loading_raw import load_raw_amr
from   amrlib.graph_processing.wiki_remover import wiki_remove_file
from   amrlib.utils.logging import silence_penman


# Load a list list of amr files as described by the list of glob patterns
# and write them to a single file.
def load_and_resave_amr_files(glob_patterns, out_fpath):
    if isinstance(glob_patterns, str):
        glob_patterns = [glob_patterns]
    # Find all the files
    fpaths = []
    for pattern in glob_patterns:
        glob_fpaths = glob(pattern)
        assert len(glob_fpaths) > 0     # check for invalid path in the list
        fpaths += glob_fpaths
    graphs = []
    # Load graphs sorted by filename for consistency
    for fpath in sorted(fpaths, key=lambda x:os.path.basename(x)):
        print('Loading', fpath)
        graphs.extend(load_raw_amr(fpath))
    print('Loaded {:,} graphs'.format(len(graphs)))
    # Save the collated data
    print('Saving data to', out_fpath)
    with open(out_fpath, 'w') as f:
        for graph in graphs:
            f.write('%s\n\n' % graph)
    print()
    return graphs


# Combine all amr3 files into dev/test and train with and without :wiki tags
def collect_amr3(base_amr_dir, out_dir):
    # Loop through the file tiles
    for amr_sub_dir in ('dev', 'test', 'training'):
        glob_pat  = os.path.join(base_amr_dir, amr_sub_dir, '*.txt')
        out_fn    = 'train.txt' if amr_sub_dir == 'training' else amr_sub_dir + '.txt'
        out_fpath = os.path.join(out_dir, out_fn)
        load_and_resave_amr_files(glob_pat, out_fpath)
        print('Removing wiki')
        wiki_remove_file(out_dir, out_fn, out_dir, out_fn + '.nowiki')
        print()


# Collect all the amr graphs from multiple files and create the dev/test/train datasets.
# This cleans out a few bad (non-ascii) characters on load and save a copy of the data
# with the :wiki tags stripped.
if __name__ == '__main__':
    silence_penman()
    base_amr_dir = 'amrlib/data/amr_annotation_3.0/data/amrs/split'
    out_dir      = 'amrlib/data/tdata_xfm'

    # Create the output directory
    os.makedirs(out_dir, exist_ok=True)
    collect_amr3(base_amr_dir, out_dir)
