#!/usr/bin/python3
import setup_run_dir    # this import tricks script to run from 2 levels up
import warnings
warnings.simplefilter('ignore')
import os
import json
from   amrlib.utils.logging import setup_logging, WARN
from   amrlib.models.parse_t5.trainer import Trainer


# Note tdata_gsii was created with 30_Model_Parse_GSII/10_Annotate_Corpus.py and 12_RemoveWikiData.py
# This can be changed.  The corpus doesn't need to be annotated (you can skip running 10_x) but
# wikidata should be removed since the model doesn't produce those tags.
if __name__ == '__main__':
    setup_logging(logfname='logs/train_t5parse.log', level=WARN)
    config_fn = 'configs/model_parse_t5.json'

    with open(config_fn) as f:
        args = json.load(f)
    trainer = Trainer(args)
    trainer.train()
