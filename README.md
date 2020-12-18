#  amrlib

**A python library that makes AMR parsing, generation and visualization simple.**


## About
amrlib is a python module designed to make processing for [Abstract Meaning Representation](https://amr.isi.edu/)
 (AMR) simple by providing the following functions
* Sentence to Graph (StoG) parsing to create AMR graphs from English sentences.
* Graph to Sentence (GtoS) generation for turning AMR graphs into English sentences.
* A QT based GUI to facilitate conversion of sentences to graphs and back to sentences
* Methods to plot AMR graphs in both the GUI and as library functions
* Training and test code for both the StoG and GtoS models.
* A [SpaCy](https://github.com/explosion/spaCy) extension that allows direct conversion of
SpaCy `Docs` and `Spans` to AMR graphs.
* Rule Based Word Alignment of tokens to graph nodes
* An evaluation metric API including including...
  - Smatch (multiprocessed with enhanced/detailed scores) for graph parsing
  - BLEU for sentence generation
  - Alignment scoring metrics detailing precision/recall
* Sentence paraphrasing - experimental


## AMR Models
The system includes two different neural-network models for parsing and one for generation.

* Parse (StoG) model **parse_t5 gives 81 SMATCH score** with LDC2020T02.  This model uses the
pretrained HuggingFace T5 transformer model to convert sentences to graph-encoded sequences which
are then deserialized into an AMR graph.

* Parse (StoG) model **parse_gsii gives 77 SMATCH score** with LDC2020T02.  This model comes from
[jcyk/AMR-gs](https://github.com/jcyk/AMR-gs), the details of which can be found in this
[paper](https://arxiv.org/abs/2004.05572).  The version of the model used here eliminates
much of the data abstraction (aka anonymization) used in the original code

* Generation (GtoS) **generate_t5 gives a BLEU score of 43** with LDC2020T02.  Similar to parse_t5,
the model takes advantage of the pretrained [HuggingFace](https://github.com/huggingface/transformers)
T5 transformer.  Details on using this type of model for generation can be found in this
[paper](https://arxiv.org/abs/2007.08426). The model is fine-tuned to translate AMR graphs to English
sentences.



## Documentation
For the latest documentation, see **[ReadTheDocs](https://amrlib.readthedocs.io/en/latest/)**.


## AMR View
The GUI allows for simple viewing, conversion and plotting of AMR Graphs.

![AMRView](https://github.com/bjascob/amrlib/raw/master/docs/images/AMRView01.png)
<!--- docs/images/AMRView01.png --->
<!--- https://github.com/bjascob/amrlib/raw/master/docs/images/AMRView01.png --->


## Requirements and Installation
The project was built and tested under Python 3 and Ubuntu but should run on any Linux, Windows, Mac, etc.. system.

See [Installation Instructions](https://amrlib.readthedocs.io/en/latest/install/) for details on setup.

## Library Usage
To convert sentences to graphs
```
import amrlib
stog = amrlib.load_stog_model()
graphs = stog.parse_sents(['This is a test of the system.', 'This is a second sentence.'])
for graph in graphs:
    print(graph)
```
To convert graphs to sentences
```
import amrlib
gtos = amrlib.load_gtos_model()
sents, _ = gtos.generate(graphs)
for sent in sents:
    print(sent)
```
For a detailed description see the [Model API](https://amrlib.readthedocs.io/en/latest/api_model/).


## Usage as a Spacy Extension
To use as an extension, you need spaCy version 2.0 or later.  To setup the extension and use it do the following
```
import amrlib
import spacy
amrlib.setup_spacy_extension()
nlp = spacy.load('en_core_web_sm')
doc = nlp('This is a test of the SpaCy extension. The test has multiple sentences.')
graphs = doc._.to_amr()
for graph in graphs:
    print(graph)
```
For a detailed description see the [Spacy API](https://amrlib.readthedocs.io/en/latest/api_spacy/).


## Paraphrasing
For an explanation of how to use the library to do paraphrasing, see the
[Paraphrasing](https://amrlib.readthedocs.io/en/latest/paraphrase/) section in the docs.


## Issues
If you find a bug, please report it on the [GitHub issues list](https://github.com/bjascob/amrlib/issues).
Additionally, if you have feature requests or questions, feel free to post there as well.  I'm happy to
consider suggestions and Pull Requests to enhance the functionality and usability of the module.
