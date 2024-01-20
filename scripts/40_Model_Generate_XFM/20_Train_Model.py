#!/usr/bin/env python3
import setup_run_dir    # this import tricks script to run from 2 levels up
import os
import json
from   amrlib.utils.logging import setup_logging, WARN
from   amrlib.models.generate_xfm.trainer import Trainer


if __name__ == '__main__':
    setup_logging(logfname='logs/train_generate_xfm.log', level=WARN)
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'    # eliminate tokenize_fast message
    config_fn = 'configs/model_generate_xfm_t5_base_wTT.json'

    with open(config_fn) as f:
        args = json.load(f)
    trainer = Trainer(args)
    trainer.train()
