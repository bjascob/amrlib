#!/usr/bin/python3
import setup_run_dir    # this import tricks script to run from 2 levels up
import os
from   amrlib.alignments.faa_aligner.trainer import Trainer


if __name__ == '__main__':
    # Specify bin directory.  If bin_dir is specified it will use this directory, otherwise
    # it w will it will look for the environment variable FABIN_DIR or use the path.
    # bin_dir     = 'xxx'  # change commented line below
    working_dir = 'amrlib/data/train_faa_aligner'

    in_fpath = os.path.join(working_dir, 'fa_in.txt')
    print('Training based on', in_fpath)
    #trainer = Trainer(in_fpath, working_dir, bin_dir=bin_dir, I=6, q=0.16, a=0.04, T=3)
    trainer = Trainer(in_fpath, working_dir, I=6, q=0.16, a=0.04, T=3)
    trainer.train()


# The above is portable python code but you can also run the following from a bash script
# If you do this manually, be sure to write the parameters to train_params.json, as this
# will get copied into amrlib_meta.json which is used by the inference TrainedAligner class.
# The Trainer class saves these params for you.
# FABIN=/usr/local/bin
# WKDIR=data/train
# $FABIN/fast_align -i $WKDIR/fa_in.txt -d -o -v -I 6 -q 0.16 -a 0.04 -T 3    -p $WKDIR/fwd_params > $WKDIR/fa_out_fwd.txt  2>$WKDIR/fwd_err
# $FABIN/fast_align -i $WKDIR/fa_in.txt -d -o -v -I 6 -q 0.16 -a 0.04 -T 3 -r -p $WKDIR/rev_params > $WKDIR/fa_out_rev.txt  2>$WKDIR/rev_err
# $FABIN/atools -i $WKDIR/fa_out_fwd.txt -j $WKDIR/fa_out_rev.txt -c grow-diag-final-and > $WKDIR/fa_out.txt
