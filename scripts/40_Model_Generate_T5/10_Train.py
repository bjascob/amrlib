#!/usr/bin/python3
import setup_run_dir    # this import tricks script to run from 2 levels up
import warnings
warnings.simplefilter('ignore')
import os
import json
from   amrlib.utils.logging import setup_logging, WARN
from   amrlib.models.generate_t5.trainer import Trainer


if __name__ == '__main__':
    setup_logging(logfname='logs/train_t5gen.log', level=WARN)
    config_fn = 'configs/model_generate_t5.json'

    with open(config_fn) as f:
        args = json.load(f)
    trainer = Trainer(args)
    trainer.train()
