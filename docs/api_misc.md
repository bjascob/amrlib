# Misc API functions


## AMR Loading
The AMR file format can be loaded in a few different ways.  One is by using the `penman.graph`
object from the [penman](https://github.com/goodmami/penman) library. This object is useful
when you want to manipulate the graphs.

For simple string loading of AMR data, there is a simple method `load_amr_entries` which can
be imported from `amrlib/graph_processing/amr_loading.py`.
```
entries = load_amr_entries(fname)
```
which returns a list of graph + metadata strings for a given filename.


## AMR Plotting
The library includes facilities to plot AMR graphs using the `graphviz` library.  The object
`AMRPlot` can be found in `amrlib/graph_processing/amr_plot.py`


### Example
```
from   amrlib.graph_processing.amr_plot import AMRPlot
from   amrlib.graph_processing.amr_loading import load_amr_entries
input_file = 'amrlib/data/LDC2020T02/test.txt'
# Load the AMR file
entries = load_amr_entries(input_file)
entry = entries[125]    # pick an index
# Plot
plot = AMRPlot()
plot.build_from_graph(entry, debug=False)
plot.view()
```
Set `debug` to `True` to print a list of triples associated with the graph.

A script for this can be found at `scripts/10_Misc/PlotAMR.py`
