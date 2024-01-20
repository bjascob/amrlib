This directory contains scripts to train and test an AMR to text model (`generate_xfm`) with and
without tense tags.

If you aren't arleady familiar with the process, you may want to read the
[ReadTheDocs page](https://amrlib.readthedocs.io/en/latest/models/#generate-models)
on this topic.

The scripts are meant to be run in order and they are geared towards a model trained to operate both
with or without tense tags present. To do this, the AMR corpus training data is shuffled and combined
with a copy of the graphs that have the tags added. This trains the model for both scenarios.  If you
want a model that is not trained with the tense tags, simply skip the `Annotate_Corpus` and
`Create_Training_Data` steps and point your configuation.json to the standard AMR training data.
You may choose to remove the wikitags using the script or simply use the unprocessed AMR corpus data.
