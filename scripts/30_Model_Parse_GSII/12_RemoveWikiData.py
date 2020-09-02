#!/usr/bin/python3
import setup_run_dir    # this import tricks script to run from 2 levels up
from   amrlib.graph_processing.wiki_remover import wiki_remove_file
from   amrlib.utils.logging import silence_penman


# Remove the :wiki tags from the graphs since we don't want to try the model
# to produce these
if __name__ == '__main__':
    silence_penman()
    data_dir = 'amrlib/data/tdata_gsii'

    # run the pipeline
    for fn in ('test.txt.features', 'dev.txt.features', 'train.txt.features'):
        wiki_remove_file(data_dir, fn, data_dir, fn + '.nowiki')
