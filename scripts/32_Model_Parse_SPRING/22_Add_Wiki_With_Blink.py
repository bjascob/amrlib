#!/usr/bin/env python3
import setup_run_dir    # Set the working directory and python sys.path to 2 levels above
import sys
# Add BLINK to python search path if needed (there is no pip install for BLINK)
sys.path.append('/home/bjascob/Libraries/BLINK-2021_12_02')  
import warnings
warnings.simplefilter('ignore')     # Blink has useless warning
import json
import penman
from   amrlib.utils.logging import setup_logging, silence_penman, WARN
from   amrlib.graph_processing.wiki_adder_blink import WikiAdderBlink


if __name__ == '__main__':
    setup_logging('logs/blink_wikify.log', level=WARN)
    silence_penman()

    model_dir  = 'amrlib/data/BLINK_Model'
    infpath    = 'amrlib/data/model_parse_spring/test-pred.txt'
    outfpath   = 'amrlib/data/model_parse_spring/test-pred.txt.wiki'

    # Load the BLINK models
    wa = WikiAdderBlink(model_dir)
    wa.wikify_file(infpath, outfpath)
