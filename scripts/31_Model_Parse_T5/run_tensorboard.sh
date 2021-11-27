#!/bin/sh
cd ../..
tensorboard --logdir amrlib/data/model_parse_t5/runs/ --host 192.168.0.103 --port 6006
