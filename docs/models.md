# Models
The following is a description of the various models and extra interface parameters they use.

## Parse T5
**81 SMATCH score** with LDC2020T02.

This model is based on the pretrained [HuggingFace](https://github.com/huggingface/transformers)
T5 transformer to convert sentences to graph-encoded sequences which are then deserialized into
an AMR graph.

Additional inference parameters:

* device     : is automatically selected but you can pass `cpu` or `cuda:0` if needed.

* batch_size : set the batch size to use for model generation

* num_beams  : set the number of beams used during beam_search (1 == greedy search)

See amrlib/models/parse_t5/inference.py for implementation details.


## Parse GSII
**77 SMATCH score** with LDC2020T02.

This model comes from [jcyk/AMR-gs](https://github.com/jcyk/AMR-gs), the details of which can be
found in this [paper](https://arxiv.org/abs/2004.05572).  The version of the model used here eliminates
much of the data abstraction (aka anonymization) used in the original code.

Additional inference parameters:

* device     : is automatically selected but you can pass `cpu` or `cuda:0` if needed.

* batch_size : set the batch size to use for model generation

* beam_size  : set the number of beams used during beam_search (1 == greedy search)

* alpha      : score = x.score/((1+len(x.seq))**alpha (default = 0.6)


See amrlib/models/parse_gsii/inference.py for implementation details.


## Generate T5wtense
**54 BLEU** with tense information (part of speech tags) added or **44 BLEU** for basic LDC2020T02 graphs.

This model is based on the pretrained [HuggingFace](https://github.com/huggingface/transformers)
T5 transformer to convert AMR Graphs to natural language sentences.

It is similar in implementation to the original generate_t5 model except that it has the
ability to add part-of-speech (POS) tags to the concepts in the AMR graph.  AMR does not normally
include any tense information so without this, the generator has no way of knowing which tense
the output sentence is supposed to be.   Adding the tags improves the overall BLEU score and creates
generated sentences that are closer to the original.

The model is trained on data with and without the tags and will create sentences with or without
tense information added.

Note that that model limit is 512 tokens (not string characters) and roughly 2.5% of the LDC2020T02
graphs need to be clipped.

Additional inference parameters:

* device     : is automatically selected but you can pass `cpu` or `cuda:0` if needed.

* batch_size : set the batch size to use for model generation

* num_beams  : set the number of beams used during beam_search (1 == greedy search)

* num_ret_seq : the number of sentences returned (must be <= num_beams)
Additional returned sentences will show up in order of score (high to low) on the returned list.
Note that a single list is returned, not a list of list. You can use `get_ans_group`  to extract
groupings if needed.

Additional generate parameters:
* use_tense  : Whether or not to add tense tags to the graph, prior to generation (default is True).
Note that the metadata tag `snt` for the original sentence, must be present for annotation.

* force_annotate : Re-annotate the graph, based on the `snt` metadata key, even if annotations exist.
Default is False.


See amrlib/models/generate_t5wtense/inference.py for implementation details.


## Generate T5
**43 BLEU** for LDC2020T02 graphs.

This model is based on the pretrained [HuggingFace](https://github.com/huggingface/transformers)
T5 transformer to convert AMR Graphs to natural language sentences.

Note that that model limit is 512 tokens (not string characters) and roughly 2.5% of the LDC2020T02
graphs need to be clipped.


Additional inference parameters:

* device     : is automatically selected but you can pass `cpu` or `cuda:0` if needed.

* batch_size : set the batch size to use for model generation

* num_beams  : set the number of beams used during beam_search (1 == greedy search)

* num_ret_seq : the number of sentences returned (must be <= num_beams)
Additional returned sentences will show up in order of score (high to low) on the returned list.
Note that a single list is returned, not a list of list. You can use `get_ans_group` to extract
groupings if needed.


See amrlib/models/generate_t5/inference.py for implementation details.
