# amrlib

**A python library that makes AMR parsing, generation and visualization simple.**


## About
amrlib is a python module designed to make processing for [Abstract Meaning Representation](https://amr.isi.edu/)
 (AMR) simple by providing the following functions

<!--- For Readthedocs, bulleted items must a space after and sub-list must have 4 spaces (and this still doesn't work) --->

* Sentence to Graph (StoG) parsing to create AMR graphs from English sentences.

* Graph to Sentence (GtoS) generation for turning AMR graphs into English sentences.

* A QT based GUI to facilitate conversion of sentences to graphs and back to sentences

* Methods to plot AMR graphs in both the GUI and as library functions

* Training and test code for both the StoG and GtoS models.

* A [SpaCy](https://github.com/explosion/spaCy) extension that allows direct conversion of
SpaCy `Docs` and `Spans` to AMR graphs.

* Rule Based Word Alignment of tokens to graph nodes

* An evaluation metric API including..., Smatch (multiprocessed with enhanced/detailed scores) for graph parsing,
BLEU for sentence generation, Alignment scoring metrics detailing precision/recall

* Sentence paraphrasing - experimental



## AMR Models

For details on available models see [models](https://amrlib.readthedocs.io/en/latest/models/).
