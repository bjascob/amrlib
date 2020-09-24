#!/usr/bin/python3
import setup_run_dir    # Set the working directory and python sys.path to 2 levels above
import logging
import penman
from   amrlib.utils.logging import setup_logging, silence_penman, DEBUG
from   amrlib.graph_processing.amr_loading import load_amr_entries
from   amrlib.alignments.rbw_aligner import RBWAligner
from   amrlib.alignments.penman_utils import strip_surface_alignments

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    setup_logging(level=DEBUG, logfname='logs/rbw_aligner_debug.log')
    silence_penman()

    fname = 'amrlib/data/alignments/test_realigned.txt'
    index = 0

    entries = load_amr_entries(fname)
    entry   = entries[index]
    entry   = strip_surface_alignments(entry)

    # Run the aligner
    aligner = RBWAligner.from_string_w_json(entry) #, align_str_name='rbw_alignments')
    print(aligner.get_graph_string())
    print()
