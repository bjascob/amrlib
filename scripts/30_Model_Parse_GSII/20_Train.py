#!/usr/bin/python3
import setup_run_dir    # this import tricks script to run from 2 levels up
from amrlib.utils.logging import setup_logging, WARN
from amrlib.utils.config import Config
from amrlib.models.parse_gsii import trainer
from amrlib.utils.log_splitter import LogSplitter


# Train th emodel
if __name__ == '__main__':
    setup_logging(logfname='logs/train_gsii.log', level=WARN)
    args = Config.load('configs/model_parse_gsii.json')
    ls = LogSplitter('train.log')
    trainer.run_training(args, ls)
