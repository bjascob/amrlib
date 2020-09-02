#!/usr/bin/python3
import setup_run_dir    # Set the working directory and python sys.path to 2 levels above
import os
from   amrlib.graph_processing.amr_loading_raw import load_raw_amr


if __name__ == '__main__':
    base_dir = 'amrlib/data/amr_annotation_3.0/data/amrs/split'
    out_dir  = 'amrlib/data/LDC2020T02'

    os.makedirs(out_dir, exist_ok=True)

    # Loop through the dirctories
    for dirname in ('dev', 'test', 'training'):
        entries = []
        dn = os.path.join(base_dir, dirname)
        print('Loading data from', dn)
        fpaths = [os.path.join(dn, fn) for fn in os.listdir(dn)]
        for fpath in fpaths:
            entries += load_raw_amr(fpath)
        print('Loaded {:,} entries'.format(len(entries)))
        # Save the collated data
        fn = 'train.txt' if dirname == 'training' else dirname + '.txt'
        out_path = os.path.join(out_dir, fn)
        print('Saving data to', out_path)
        with open(out_path, 'w') as f:
            for entry in entries:
                f.write('%s\n\n' % entry)
        print()
