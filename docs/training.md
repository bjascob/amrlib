# Training and Test
The library includes scripts to prepare training data and to train both the parse and generate
models.  These scripts are not part of the `pip` install so to use them simply download the
GitHub project and use them in place.  The code for these is located in `scripts/X_yyy`.

# Scripts Directory
```
* 10_Misc                  PlotAMR.py, SpotlightDBServer.sh
* 20_Assemble_LDC2020T02   Collect the LDC training data into test, dev, and train files
* 30_Model_Parse_GSII      Scripts to train the parsing model
* 40_Model_Generate_T5     Scripts to train the generation model
* 50_Build_AMR_View        Script to build and run the GUI
```
Most of the files in these directories start with a number.  To train the model, open the files
in numerical order, check the defined directories, etc.. under the `main` statement and run.
Most of these scripts are simply there to setup parameters and then call the associated library
functions to execute the code.  As such, they are all relatively short and should be fairly
self-explanatory.

## Directory Structure and Run Locations
The scripts in the above directories have a statement at the top `import setup_run_dir`.  This
very simple import causes python to see the script as running from 2 levels up from the current
directory.  This is just a simple way to keep the scripts all in an organized directory and still
have local import of `amrlib` and a common path to `data`.  If you move the script or try to run
it from another location, be sure to remove this import and modify paths accordingly.

All training data, models, etc.. is expected to reside under `amrlib/data`.  When you run the first
script for data prep (`scripts/20_Assemble_LDC2020T02/10_CollateData.py`) the only directory present
should be `amrlib/data/amr_annotation_3.0`.  After running all the scripts in 20_, 30_ and 40_ the
data directory layout will look something like...
```
├── amr_annotation_3.0
├── LDC2020T02
│   ├── dev.txt
│   ├── test.txt
│   └── train.txt
├── model_generate_t5
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── test.txt.generated
│   ├── test.txt.ref_sents
│   └── training_args.bin
├── model_parse_gsii
│   ├── epoch200.pt
│   ├── epoch200.pt.dev_generated
│   ├── epoch200.pt.test_generated
│   ├── epoch200.pt.test_generated.wiki
│   └── vocabs
│       ├── concept_char_vocab
│       ├── concept_vocab
│       ├── lem_char_vocab
│       ├── lem_vocab
│       ├── ner_vocab
│       ├── pos_vocab
│       ├── predictable_concept_vocab
│       ├── rel_vocab
│       ├── tok_vocab
│       └── word_char_vocab
└── tdata_gsii
    ├── dev.txt.features
    ├── dev.txt.features.nowiki
    ├── spotlight_wiki.json
    ├── test.txt.features
    ├── test.txt.features.nowiki
    ├── train.txt.features
    └── train.txt.features.nowiki
```
Note that when downloading models you will get a similar layout but the model directories generally
have a version number appended and a link set, ie.. `model_stog -> model_parse_gsii-v0_1_0`.


## Model Configuration
The `configs` directory has `json` files that contain model parameters used for training.  You will
notice that the train scripts for both models load these.  If you wnat to change the location of
the training data or any other model / training parameters (such as batch size) check in these
files.


## Training data
The latest AMR training corpus, [LDC2020T02](https://catalog.ldc.upenn.edu/LDC2020T02), is available
from the Linguistic Data Consortium.  It is free for institutions that have a membership or $300
for non-members (for non-commercial use).

This newest corpus contains about 60K AMR graphs.
Other versions of LDC data can be used for training and test, however earlier versions are generally
smaller so expect SMATCH and BLEU scores to be slightly lower on the smaller datasets.  The original, freely
available, "Little Prince" corpus is much smaller that the LDC datasets.  It is not big enough to do
a good job of training these large models but it can be used for experimenting; just expect much lower
scores during test.

If you want to try training but don't want to buy the LDC data, it's reasonable to use the pre-trained parser
(or another existing one such as [JAMR](https://github.com/jflanigan/jamr)) to create a synthetic corpus
by parsing a large number of sentences from a free corpus and then using the output AMR graphs as input for training.
This technique has shown to be an effective pre-training method in some papers, however with the larger
LDC2020T02 corpus, pre-training is not generally required.

If you change the name / location of the training files, be sure to update the associated `.json` config
files.
