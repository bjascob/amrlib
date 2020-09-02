#!/usr/bin/python3
import setup_run_dir    # Set the working directory and python sys.path to 2 levels above
import json
from   PyQt5.QtWidgets import *
from   amrlib.amr_view.main_window import MainWindow
from   amrlib.utils.logging import silence_penman, setup_logging, INFO, WARN


if __name__ == '__main__':
    setup_logging(level=WARN)
    silence_penman()

    with open('amrlib/amr_view/amr_view.json') as f:
        config = json.load(f)

    app = QApplication([])
    window = MainWindow(config)
    app.exec_()
