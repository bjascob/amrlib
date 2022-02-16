#!/usr/bin/python3
import setup_run_dir    # Set the working directory and python sys.path to 2 levels above
import os
import json
from   copy import deepcopy
from   amrlib.utils.logging import setup_logging, WARN
from   amrlib.evaluate.smatch_enhanced import redirect_smatch_errors
from   amrlib.models.parse_xfm.trainer import Trainer


if __name__ == '__main__':
    setup_logging(logfname='logs/train_model_parse_xfm.log', level=WARN)
    redirect_smatch_errors('logs/train_smatch_errors.log')
    config_fn = 'configs/model_parse_xfm_bart_base.json'        # Choose the appropriate config file here

    #os.environ['CUDA_VISIBLE_DEVICES']   = '0'     # select the GPUs to use
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    with open(config_fn) as f:
        config = json.load(f)

    trainer = Trainer(config)
    trainer.train()
