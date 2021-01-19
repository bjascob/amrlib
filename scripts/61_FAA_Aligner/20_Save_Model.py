#!/usr/bin/python3
import setup_run_dir    # this import tricks script to run from 2 levels up
import os
import json
import tarfile
from   shutil import copyfile


# amrlib_meta.json data for the model
meta = {"model_name":"faa_aligner",
        "model_type":"span_aligner",
        "version":"0.1.0",
        "date":"2021-01-18",
        "fa_commit_data":"2020-01-08",
        "inference_module":".faa_aligner.faa_aligner",
        "inference_class":"FAA_Aligner",
        "kwargs":{}
}


if __name__ == '__main__':
    train_dir = 'amrlib/data/train_faa_aligner'
    model_dir = 'amrlib/data/model_aligner_faa'
    tar_fn    = 'amrlib/data/train_faa_aligner/model_aligner_faa.tar.gz'

    # Create the directory and copy a copy of files
    print('Copying parameter data to', model_dir)
    os.makedirs(model_dir, exist_ok=True)
    copyfile(os.path.join(train_dir, 'fwd_params'), os.path.join(model_dir, 'fwd_params'))
    copyfile(os.path.join(train_dir, 'rev_params'), os.path.join(model_dir, 'rev_params'))

    # Merge the train_params.json with the above metadata and write to the model directory
    with open(os.path.join(train_dir, 'train_params.json')) as f:
        train_params = json.load(f)
    meta['train_params'] = train_params
    with open(os.path.join(model_dir, 'amrlib_meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)

    # Create a tar.gz file
    print('Creating', tar_fn)
    with tarfile.open(tar_fn, 'w:gz') as tar:
        tar.add(model_dir, arcname=os.path.basename(model_dir))
