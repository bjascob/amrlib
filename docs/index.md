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
The system uses two different Neural Network models for parsing and generation.

The parsing (StoG) model comes from [AMR-gs](https://github.com/jcyk/AMR-gs), the details of which
can be found in this [paper](https://arxiv.org/abs/2004.05572).  The version of the model used here eliminates
much of the data abstraction (aka anonymization) used in the original code.  During testing, this model
achieves a 77 SMATCH score with LDC2020T02.

The generation (GtoS) model takes advantage of the pretrained [HuggingFace](https://github.com/huggingface/transformers)
T5 transformer.  The model is fine-tuned to translate AMR graphs to English sentences.  The retrained model
achieves a BLEU score of 43 with LDC2020T02.
