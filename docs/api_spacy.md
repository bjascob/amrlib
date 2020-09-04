# Spacy API
The system can operate as a pipeline add-in to spaCy that attaches to a `span` or `doc`
To use as an extension, you need spaCy version 2.0 or later.


## Functions
The function `setup_spacy_extension()` attaches a seqeuence-to-graph inference function to the
spaCy `span` and `doc` objects.  When working with these objects, there is a new method `<object>._.to_amr()`.
Note the `._.` that spaCy uses to delineate add-on functions from native spaCy functions for
these objects.


### Example
```
import amrlib
import spacy
amrlib.setup_spacy_extension()
nlp = spacy.load('en_core_web_sm')
doc = nlp('This is a test of the SpaCy extension. The test has multiple sentences.')

# The following are roughly equivalent but demonstrate the different objects.
graphs = doc._.to_amr()
for graph in graphs:
    print(graph)

for span in doc.sents:
    graphs = span._.to_amr()
    print(graphs[0])
```
