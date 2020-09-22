import os
import sys
import time
import traceback
import logging
from   PyQt5.QtGui import QFontDatabase
from   PyQt5.QtWidgets import QMainWindow, QFileDialog, QInputDialog
from   PyQt5.QtCore import QTimer
from   .Ui_MainWindow import Ui_MainWindow
from   .processor_stog import ProcessorSTOG
from   .processor_gtos import ProcessorGTOS
from   ..graph_processing.amr_plot import AMRPlot
from   ..graph_processing.amr_loading import load_amr_entries

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, config, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.config = config
        # Setup the UI
        self.setupUi(self)
        fixed_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        fixed_font.setPointSize(8)
        #self.inputSentLE.setFont(fixed_font)
        #self.generatedTE.setFont(fixed_font)
        self.amrTE.setFont(fixed_font)
        self.show()
        # Setup the signals/slots for buttons
        self.exitPB.pressed.connect(self.exit_slot)
        self.toAmrPB.pressed.connect(self.parse_slot)
        self.generatePB.pressed.connect(self.generate_slot)
        self.showGraphPB.pressed.connect(self.show_graph_slot)
        # Signals/slots for menu itesm
        self.actionLoadAMR.triggered.connect(self.load_amr_slot)
        self.actionSaveAMR.triggered.connect(self.save_amr_slot)
        self.last_open = '.'
        self.last_save = '.'
        # Create the processors
        self.stog_processor = ProcessorSTOG(self.config, disabled=False)  # Disable for debug
        self.gtos_processor = ProcessorGTOS(self.config, disabled=False)
        # Set the main window title  / status with a qtimer
        self.startup_timer = QTimer(self)
        self.startup_timer.setInterval(250)     # ms
        self.startup_timer.timeout.connect(self.set_window_title)
        self.startup_timer.start()

    def set_window_title(self):
        if self.stog_processor.is_ready() and self.gtos_processor.is_ready():
            self.setWindowTitle('AMR Visualization - Ready')
            self.startup_timer.stop()
        else:
            self.setWindowTitle('AMR Visualization - Loading Models')

    def exit_slot(self):
        sys.exit(0)

    def parse_slot(self):
        if not self.stog_processor.is_ready():
            return
        text = self.inputSentLE.text().strip()
        try:
            amr_string = self.stog_processor.run(text).strip()
            self.amrTE.setPlainText(amr_string)
        except:
            self.amrTE.setPlainText('\nERROR - Not able to parse.')
            traceback.print_exc()

    def generate_slot(self):
        if not self.gtos_processor.is_ready():
            return
        text = self.amrTE.toPlainText().strip()
        try:
            string = self.gtos_processor.run(text)
            self.generatedTE.setPlainText(string)
        except:
            self.generatedTE.setPlainText('\nERROR - Not able to generate.')
            traceback.print_exc()

    def show_graph_slot(self):
        text = self.amrTE.toPlainText().strip()
        # Generate first
        if not text:
            self.parse_slot()
            text = self.amrTE.toPlainText().strip()
            if not text:
                logger.warning('No graph to plot')
                return
        try:
            render_fn = self.config.get('render_fn', None)
            format    = self.config.get('render_format', 'pdf')
            plot = AMRPlot(render_fn, format)
            plot.build_from_graph(text)
            plot.view()
        except:
            logger.warning('Exception when trying to plot')
            traceback.print_exc()

    def load_amr_slot(self):
        # Get the filename from the file dialog
        dlg = QFileDialog()
        dlg.resize(100, 100)
        path, _ = dlg.getOpenFileName(self, "Open file", self.last_open, "TXT (*.txt);; (*.*)")
        self.last_open = path
        if not path:
            return
        # Load the entire file
        try:
            entries = load_amr_entries(path)
        except:
            logger.warning('Unable to load %s' % path)
        # Find which entry to display
        num, ok = QInputDialog().getInt(self, 'Which entry to load', 'Entry number')
        if not ok: return
        # Load and display the entry
        num = max(0, min(num, len(entries)-1))  # clip possible values
        entry = entries[num]
        self.inputSentLE.clear()
        self.generatedTE.clear()
        self.amrTE.setPlainText(entry)

    def save_amr_slot(self):
        # Get the filename
        path, _ = QFileDialog.getSaveFileName(self, "Save to file", self.last_save, "TXT (*.txt);; All (*.*)")
        self.last_save = path
        if not path:
            return
        text = self.amrTE.toPlainText().strip()
        if not text:
            logger.warning('No AMR to save')
            return
        with open(path, 'w') as f:
            f.write(text + '\n')
        logger.info('File saved to %s' % path)
