# Installation Instructions

The project was built and tested under Python 3 and Ubuntu but should run on any Linux, Windows, Mac, etc.. system.


## Install the code

* Install pytorch using the [instructions](https://pytorch.org/) specific to your machine setup. A GPU/cuda is not required
for run-time use but is highly recommended for training models.

* If you want to plot graphs, follow the graphviz installation instructions on the [pypi page](https://pypi.org/project/graphviz/).
This requires both the pip graphviz install and the installation of the non-python Graphviz library.  The separate installs
are required because graphviz is a python wrapper for Graphviz which pip can't install by itself.

* If you want to run the faa_aligner, you will need to install and compile [fast_align](https://github.com/clab/fast_align).
Put this in your path or you can set the environment variable `FABIN_DIR` to its location.

`pip3 install -r requirements.txt`

`python3 -m spacy download en_core_web_sm`

`pip3 install amrlib`

Note that installing amrlib will automatically install a minimal set of requirements but for the QT based amr_view
or to test/train a model, you'll need to also install from the requirements.txt file.

If you want to use the FAA_Aligner, you will need to will need to compile and install the C++ code for
[fast_align](https://github.com/clab/fast_align).  The compile process will produce binaries for `fast_align`
and `atools` in the same directory. These needs to be in your path or alternately you can set the environment
variable `FABIN_DIR` to their directory.  The aligner/fast_align binaries work under both Windows and Linux.

If you want to use a different spaCy model for parsing, you can manually change the model in amrlib/defaults.py.

Note that the goal is to keep amrlib compatible with the latest versions of 3rd party libraries, however if a problem occurs you can
review the file [req_tested_versions.txt](https://github.com/bjascob/amrlib/blob/master/req_tested_versions.txt) to see
what versions were tested when the library was last released.


## Install the models

Download links for all the models can be found at [amrlib-models](https://github.com/bjascob/amrlib-models).

These files need to be extracted and reside in the install directory under `amrlib/data` and should be named
`model_stog` (for the parse model) and `model_gtos` (for the generate model).  These will be the default models
loaded which you do `stog = amrlib.load_stog_model()`.  If you want to have multiple models of the same type on
your system, you'll need to supply the directory name when loading.  ie..
`stog = amrlib.load_stog_model(model_dir='amrlib/data/model_parse_t5-v0_1_0')`

If you're unsure what directory
amrlib is installed in you can do
```
pip3 show amrlib
```
```
>>> import amrlib
>>> amrlib.__file__
```
On a Linux system it is probably easiest to set a link to these files.  To do this, do something like..
```
cd <xx>/amrlib/data

tar xzf model_parse_gsii-v0_1_0.tar.gz
ln -snf model_parse_gsii-v0_1_0    model_stog

tar xzf model_generate_t5-v0_1_0.tar.gz
ln -snf model_generate_t5-v0_1_0   model_gtos
```
If you are on a Windows system you can simply rename the directories if this is easier than linking.
The [7-zip](https://www.7-zip.org/) utility is a popular program for extracting tar.gz files under Windows.

Note that the first time a model is used (`stog.parse_sents()` or `gtos.generate()`) the Huggingface pretrained
base models and tokenizers will automatically download. These will be cached and will not be re-downloaded
after that.


## For Training

The code base also includes library functions and scripts to train and test the parsing and generation nets.
The scripts to do this are included in the scripts directory which is not part of the pip installation.
If you want to train the networks, it is recommended that you download or clone the source code and use it in-place.
