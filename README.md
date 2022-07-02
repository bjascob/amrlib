#  amrlib

**A python library that makes AMR parsing, generation and visualization simple.**

For the latest documentation, see **[ReadTheDocs](https://amrlib.readthedocs.io/en/latest/)**.

**!! Note:** The models must be downloaded and installed separately.  See the [Installation Instructions](https://amrlib.readthedocs.io/en/latest/install).

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
* Sentence to Graph alignment routines
  - FAA_Aligner (Fast_Align Algorithm), based on the ISI aligner code detailed in this
    [paper](https://www.isi.edu/~damghani/papers/amr_eng_align.pdf).
  - RBW_Aligner (Rule Based Word) for simple, single token to single node alignment
* An evaluation metric API including including...
  - Smatch (multiprocessed with enhanced/detailed scores) for graph parsing
  - BLEU for sentence generation
  - Alignment scoring metrics detailing precision/recall



## AMR Models
The system includes different neural-network models for parsing and for generation. **!! Note:** Models must be downloaded and installed separately.
See [amrlib-models](https://github.com/bjascob/amrlib-models) for all parse and generate model download links.

* Parse (StoG) model_parse_xfm_bart_large gives an **83.7 SMATCH score** with LDC2020T02.

* Generation (GtoS) generate_t5wtense gives a **54 BLEU** with tense tags or **44 BLEU** with un-tagged LDC2020T02.


## AMR View
The GUI allows for simple viewing, conversion and plotting of AMR Graphs.

![AMRView](https://github.com/bjascob/amrlib/raw/master/docs/images/AMRView01.png)
<!--- docs/images/AMRView01.png --->
<!--- https://github.com/bjascob/amrlib/raw/master/docs/images/AMRView01.png --->


## AMR CoReference Resolution
The library does not contain code for AMR co-reference resolution but there is a related project
at [amr_coref](https://github.com/bjascob/amr_coref).

The following papers have GitHub projects/code that have similar or better scoring than the above..
* [VGAE as Cheap Supervision for AMR Coreference Resolution](https://github.com/IreneZihuiLi/VG-AMRCoref)
* [End-to-end AMR Coreference Resolution](https://github.com/Sean-Blank/AMRcoref)




## Requirements and Installation
The project was built and tested under Python 3 and Ubuntu but should run on any Linux, Windows, Mac, etc.. system.

See [Installation Instructions](https://amrlib.readthedocs.io/en/latest/install) for details on setup.

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
For an example of how to use the library to do paraphrasing, see the
[Paraphrasing](https://amrlib.readthedocs.io/en/latest/paraphrase/) section in the docs.


## Issues
If you find a bug, please report it on the [GitHub issues list](https://github.com/bjascob/amrlib/issues).
Additionally, if you have feature requests or questions, feel free to post there as well.  I'm happy to
consider suggestions and Pull Requests to enhance the functionality and usability of the module.
