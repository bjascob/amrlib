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


## AMR Models
The system uses two different Neural Network models for parsing and generation.

The parsing (StoG) model comes from [AMR-gs](https://github.com/jcyk/AMR-gs), the details of which
can be found in this [paper](https://arxiv.org/abs/2004.05572).  The version of the model used here eliminates
much of the data abstraction (aka anonymization) used in the original code.  During testing, this model
achieves a 77 SMATCH score with LDC2020T02.

The generation (GtoS) model takes advantage of the pretrained [HuggingFace](https://github.com/huggingface/transformers)
T5 transformer.  The model is fine-tuned to translate AMR graphs to English sentences.  The retrained model
achieves a BLEU score of 43 with LDC2020T02.


## Documentation
For the latest documentation, see **[ReadTheDocs](https://amrlib.readthedocs.io/en/latest/)**.


## AMR View
The GUI allows for simple viewing, conversion and plotting of AMR Graphs.

![AMRView](https://github.com/bjascob/amrlib/raw/master/docs/images/AMRView01.png)
<!--- docs/images/AMRView01.png --->
<!--- https://github.com/bjascob/amrlib/raw/master/docs/images/AMRView01.png --->

## Requirements and Installation
The project was built and tested under Python 3 and Ubuntu but should run on any Linux, Windows, Mac, etc.. system.

To install do..

`pip3 install -r requirements.txt`

`pip3 install amrlib`

Note that installing amrlib will automatically install a minimal set of requirements but for the QT based amr_view
or to test/train a model, you'll need to also install from the requirements.txt file.


To install the pretrained models do..
```
import amrlib
amrlib.download('model_stog', stog_url)    <-- stog_url below
amrlib.download('model_gtos', gtos_url)    <-- gtos_url below

stog_url =
'https://p-def8.pcloud.com/cBZLUUPPBZBfnwosZZZT4y137Z2ZZe3VZkZYRHagZC7ZVpZBHZyFZ9pZI0ZhXZU7ZYZdkZh7ZrFZCFZIpZD2z0XZTS00VM2QHM4XvD8cvftRmB8ghiTk/model_parse_gsii-v0_1_0.tar.gz'

gtos_url =
'https://p-def5.pcloud.com/cBZ9VvYPBZ56xFosZZZCLy137Z2ZZe3VZkZ2LOTcZzVZ40ZlkZVHZBFZu0ZaJZnJZEpZP5Z4pZokZcJZuJZF2z0XZtjveznPmwmm9KNc7cg0rRurX0Lnk/model_generate_t5-v0_1_0.tar.gz'
```
!! If you're get errors trying to download the models, email me at bjascob@msn.com and I'll email back a link from pcloud that you can use to download them manually.


The code base also includes library functions and scripts to train and test the parsing and generation nets.
The scripts to do this are included in the scripts directory which is not part of the pip installation.
If you want to train the networks, it is recommended that you download or clone the source code and use it in-place.


## Library Usage
To convert sentences to grahs
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
sents, _ = gtos.generate(graphs, disable_progress=True)
for sent in sents:
    print(sent)
```
For a detailed description see the [User API](https://amrlib.readthedocs.io/en/latest/api/).


## Usage as a Spacy Extension
To use as an extension, you need spaCy version 2.0 or later.  To setup the extension and use it do the following
```
import amrlib
import spacy
amrlib.setup_spacy_extension()
nlp = spacy.load('en')
doc = nlp('This is a test of the SpaCy extension. The test has multiple sentences.')
graphs = doc._.to_amr()
for graph in graphs:
    print(graph)
```

## Issues
If you find a bug, please report it on the [GitHub issues list](https://github.com/bjascob/amrlib/issues).
Additionally, if you have feature requests or questions, feel free to post there as well.  I'm happy to
consider suggestions and Pull Requests to enhance the functionality and usability of the module.
