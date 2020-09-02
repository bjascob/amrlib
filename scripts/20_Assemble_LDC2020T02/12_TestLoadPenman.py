#!/usr/bin/python3
import setup_run_dir    # Set the working directory and python sys.path to 2 levels above
import os
import penman


if __name__ == '__main__':
    data_dir  = 'amrlib/data/LDC2020T02'

    for fn in ('dev.txt', 'test.txt', 'train.txt'):
        fpath = os.path.join(data_dir, fn)
        print('Loading', fpath)
        graphs = penman.load(fpath)
        print('Loaded {:,} graphs'.format(len(graphs)))
        print()
