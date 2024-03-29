# Paraphrasing Example
The following is an explanation on how to use the library to try paraphrasing sentences.

## Process
Once you have an AMR graph, sentences are generated by passing the graph string to the T5 based
sentence generator to "translate" the graph to a sentence.

The graphs themselves have a specified root or `top`.  Changing the graph's `top` variable will change the
order of how it is serialized into a string and that re-ordering will impact the generated sentence.

The AMR spec refers to the graph's root as its [focus](https://github.com/amrisi/amr-guidelines/blob/master/amr.md#focus).

The `penman` library allows an easy method to change the graph's `top`. The following is a simple example
of how to do this...
```
pgraph = penman.decode(graph_string)
pgraph.metadata = {}    # exclude metadata from the string
new_graph_string = penman.encode(pgraph, top=new_top)
gen_sents, _ = inference.generate([new_graph_string])
```
To choose the new_top variable you can look at...
```
print('Existing top', pgraph.top)
print('tops:', pgraph.variables())    # candidates for 'top'
for triple in pgraph.instances():
    print('top=%s  concept=%s' % (triple.source, triple.target))
```
When you try this, you can see that the choice of the top variable impacts sentence generation.
Some sentences will no longer be well formed but others will have re-ordered wordings that paraphrase
the original sentence.

A more complete example that loops through all possible top variables in a graph is available in the scripts directory as [30_Paraphrasing_Example.py](https://github.com/bjascob/amrlib/blob/master/scripts/40_Model_Generate_XFM/30_Paraphrasing_Example.py).
