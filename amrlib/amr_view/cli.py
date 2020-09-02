#!/usr/bin/python3
import os
import json
import logging
from   PyQt5.QtWidgets import *
from   amrlib.amr_view.main_window import MainWindow
from   amrlib.utils.logging import silence_penman, setup_logging, INFO, WARN


# Set the default config file
config_fn = os.path.join(os.path.dirname(__file__), 'amr_view.json')
base_dir  = os.path.join(os.path.dirname(__file__), '..', '..')

# Command line interface for amr_view GUI
# setup.py should generate an exe for this by adding something like...
#   entry_points={ "gui_scripts": ['amr_view = amrlib.amr_view:cli.main']}
def main():
    setup_logging(level=WARN)
    silence_penman()

    # Open the config file
    with open(config_fn) as f:
        config = json.load(f)

    # Modify model paths to be absolute, relative to this file
    config['gtos_model_dir'] = os.path.realpath(os.path.join(base_dir, config['gtos_model_dir']))
    config['stog_model_dir'] = os.path.realpath(os.path.join(base_dir, config['stog_model_dir']))

    # For debug
    print('AMRView Config')
    for k, v in config.items():
        print('%s = %s' % (k, v))

    app = QApplication([])
    window = MainWindow(config)
    app.exec_()
