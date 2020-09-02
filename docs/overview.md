# amrlib

**A python library that makes AMR parsing, generation and visualization simple.**


## About
amrlib is a python module designed to make processing for Abstract Meaning Representation (AMR) simple
by providing the following functions
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

The code base also includes library functions and scripts to train and test the parsing and generation nets.
The scripts to do this are included in the scripts directory which is not part of the pip installation.
If you want to train the networks, it is recommended that you download or clone the source code and use it in-place.
