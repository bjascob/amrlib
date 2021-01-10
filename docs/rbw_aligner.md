# Rule Based Word Aligner

This is a simple word aligner based on the [JAMR Word Aligner](https://github.com/jflanigan/jamr/blob/Semeval-2016/src/AlignWords.scala).
It does not include the more complicated alignment of spans (aka phrases) which is the default JAMR
alignment method and it doesn't align tokens to multiple graph nodes as the ISI aligner does.

### Usage
To use the aligner you must have a graph, in either string or penman format, annotated in json format
with metadata for `tokens` and `lemmas`.  If you have a non-annotated graph string, you can use
`add_lemmas` to create the required metadata.

Example annotation (use if needed, some graphs are already properly annotated)
```
from amrlib.graph_processing.annotator import add_lemmas
penman_graph = add_lemmas(graph_string, snt_key='snt')
```

Example aligner usage
```
from amrlib.alignments.rbw_aligner import RBWAligner
aligner = RBWAligner.from_penman_w_json(penman_graph)    # use this with an annotated penman graph object
graph_string  = aligner.get_graph_string()               # get the aligned graph string
print(graph_string)
```
```
from amrlib.alignments.rbw_aligner import RBWAligner
aligner = RBWAligner.from_string_w_json(graph_string)  # use this with a graph string that is properly annotated
penman_graph = aligner.get_penman_graph()              # get the aligned penman graph object
```

See the [RBW_Aligner scripts directory](https://github.com/bjascob/amrlib/tree/master/scripts/60_RBW_Aligner)
for a number of scripts related using and testing the aligner.



## Performance
Score of RBW Aligner alignments against gold ISI hand alignments for LDC2014T12 <sup>**1</sup>
```
* dev set:    Precision: 88.82   Recall: 55.72   F1: 68.48
* test set:   Precision: 90.36   Recall: 58.77   F1: 71.22
```

The high precision, low recall score indicates that most of the alignments RBW produces match up to
an alignment in the gold set but the gold set has a fair number more alignments.  This is because
the RBW aligner is attempting to match each word in the token list to a single node or edge in
the graph. Its logic prevents it from matching a word to more than one item in the graph.
The gold alignments match multiple items in the graph to words in the token list.  This does not
require a 1:1 relationship.  A token can, and often does, match to multiple graph items.

As an example..

Tokens: "Xinhua News Agency , Beijing , September 1 st , by reporter Guojun Yang"
![alignment meld](https://github.com/bjascob/amrlib/raw/master/docs/images/rbw_vs_isi_alignments_example01.png)
<!--- docs/images/rbw_vs_isi_alignments_example01.png--->
<!--- https://github.com/bjascob/amrlib/raw/master/docs/images/rbw_vs_isi_alignments_example01.png --->

For the ISI alignment above we have `:month~e.6 9~e.6` so both the edge and it's literal value are aligned.
For the RBW aligner it has `:month 9~e.6` where only the literal is aligned.  This behavior is common
and accounts for the low recall score with the RBW aligner.

<sup>**1</sup>
ISI hand alignments can be downloaded from http://www.isi.edu/natural-language/mt/dev-gold.txt and
http://www.isi.edu/natural-language/mt/test-gold.txt.  These are each 100 alignments for the 100
AMR entries in the "dev-consensus.txt" and "test-consensus.txt" files in the amr 1.0 split directories.
