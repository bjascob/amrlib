#!/usr/bin/python3
import setup_run_dir    # this import tricks script to run from 2 levels up
from   amrlib.graph_processing.amr_plot import AMRPlot
from   amrlib.graph_processing.amr_loading import load_amr_entries


if __name__ == '__main__':
    input_file = 'amrlib/data/LDC2020T02/test.txt'
    snum = 4       # id numbers start at 1 so they are 1 more than snum

    # Load the file
    entries = load_amr_entries(input_file)
    print('Found %d entries' % len(entries))
    print()

    # Parse AMR
    entry = entries[snum]
    plot = AMRPlot()
    plot.build_from_graph(entry, debug=False)
    plot.view()
