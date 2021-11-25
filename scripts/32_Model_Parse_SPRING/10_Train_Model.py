#!/usr/bin/env python3
import setup_run_dir    # this import tricks script to run from 2 levels up
import json
import random
import logging
import torch
import numpy
from   amrlib.models.parse_spring.trainer import Trainer
from   amrlib.utils.logging import setup_logging, silence_penman, WARN

# See random generators for consistant results
random.seed(0)
torch.manual_seed(0)
numpy.random.seed(0)


# For bart-large
#   There are ~16068 batches in the training data for batch_size = 500
#   On a Titan X (12GB, fp32=6.7 TFlops) training takes 80 minutes/epoch including
#   about  6 minutes for prediction/smatch testing.
if __name__ == '__main__':
    logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)   # skip tokenizer warning
    setup_logging(logfname='logs/train_parse_spring.log', level=WARN)
    silence_penman()

    # Paths
    config_fn  = 'configs/model_parse_spring.json'
    #checkpoint = 'data/model_parse_spring/checkpoint_epoch_08_smatch_8422.pt'
    checkpoint = None   # start from scratch

    # Load the config file
    with open(config_fn) as f:
        config = json.load(f)

    # Setup the training data locations
    config['train'] = 'amrlib/data/amr_annotation_3.0/data/amrs/split/training/*.txt'
    config['dev']   = 'amrlib/data/amr_annotation_3.0/data/amrs/split/dev/*.txt'

    # Run the training
    trainer = Trainer(config)
    trainer.train(checkpoint=checkpoint)
