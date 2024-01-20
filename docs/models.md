# Models
The following is a description of the various models and extra interface parameters they use.


## Parse Models

There are several software modules for converting English sentences to AMR graphs.
These include `parse_gsii`, `parse_spring` and `parse_xfm`. The `parse_gsii` and `parse_spring` are
3rd party modules with their model specific code.  The `parse_xfm` (aka parsing transformer) module was
created as part of amrlib and is generally the best performing.  This module uses pretrained Sequence-to-Sequence
models from the Huggingface transformers library that that have been fine-tuned for parsing in amrlib.
See [amrlib-models](https://github.com/bjascob/amrlib-models) for several different models, based off of
various pretrained models such as bart and t5.

Note that the new `parse_xfm` module replaces the older `parse_t5` code. Older t5 parse models
will still work with this newer code.

To convert sentences to graphs use the following example code.  All modules have a the same basic API. The `load_stog_model()` will automatically select the proper software module to use, based off the model's `amrlib_meta.json` file.
```
import amrlib
stog = amrlib.load_stog_model()
graphs = stog.parse_sents(['This is a test of the system.', 'This is a second sentence.'])
for graph in graphs:
    print(graph)
```

Parameters for `load_stog_model()`:

* model_dir : by default the code loads the model at amrlib/data/model_stog. Alternately you can specify the path here.

* device : is automatically selected but you can pass `cpu` or `cuda:0` if needed.

* batch_size : set the batch size to use for model generation (default is 12)

* num_beams : set the number of beams used during beam_search. 1 is a greedy search. (default is 4).


Parameters for `parse_sents()`:

* add_metadata : Add `# ::snt <sentence>` to the returned graphs (default it True)

* disable_progress : disable the tqdm progress indictor (default is True)



## Generate Models

The module for AMR graph to English sentence generation is named `generate_xfm` (generting transformer).
As with the parse models, the generate model makes use of pretrained models such as bart and t5.
The training code is used to fine-tune them for the graph to sentence task. Practically, these two operations are the inverse of one another and training the transformer simply
involves swapping the setences from outputs to inputs in the training data.

The one specific feature of the generation task is to allow for the addition of part-of-speech (POS) tags
to the concepts in the AMR graph.  AMR does not normally include any tense information so without this,
the generator has no way of knowing which tense the output sentence is supposed to be.  Adding the tags
improves the overall BLEU score and creates generated sentences that are closer to the original.

The code allows training a model with or without these tags present and the user can choose whether to use
the feature or not, when running inference on the model.

Note that the new `generate_xfm` module replaces the previous `generate_t5` and `generate_t5wtense` modules. Older t5 models will still work with the newer code.

To convert graphs to sentences us the following code. The `load_gtos_model()` will automatically call the correct software module based on the model's `amrlib_meta.json` file.
```
import amrlib
gtos = amrlib.load_gtos_model()
sents, _ = gtos.generate(graphs)
for sent in sents:
    print(sent)
```

Parameters for `load_gtos_model()`

* model_dir  : by default the code loads the model at amrlib/data/model_gtos. Alternately you can specify the path here.

* device     : is automatically selected but you can pass `cpu` or `cuda:0` if needed.

* batch_size : set the batch size to use for model generation. (default is 32)

* num_beams  : set the number of beams used during beam_search 1 => greedy search (default is 4).

* num_ret_seq : the number of sentences returned (must be <= num_beams) (default is 1).
Additional returned sentences will show up in order of score (high to low) on the returned list.
Note that a single list is returned, not a list of list. You can use `get_ans_group`  to extract
groupings if needed.


Parameters for `generate()`

* use_tense  : Whether or not to add tense tags to the graph, prior to generation (default is True).
The processs requires aligning the graph to the sentence so the metadata field `snt` needs to be present.
By default, the method will use the `pos_tag` field in the graph's metadata as the list of tags to apply
once aligned. If this field is not present, it will be generated from the `snt` tag. If neither are present,
no tense tags will be applied (see logger output for failures).

* reannotate : Re-annotate the graph, based on the `snt` metadata key, even if annotations exist. (default is False)

* disable_progress : disable the tqdm progress indictor (default is True)
