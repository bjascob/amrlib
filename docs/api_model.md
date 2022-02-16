# Inference Model API
This documents the basic API for the system.  The methods here just a small subset of
the various functions in the system but should represent the majority of what's needed for
run-time (aka inference) operation.

For training and test examples see [Training](https://amrlib.readthedocs.io/en/latest/training/)
or refer directly to the code.

For additional examples, see the `scripts/xx` directories or the `tests/xx` directories.


## Sequence to Graph Functions (Parsing)
**load_stog_model()**
```
stog = load_stog_model(model_dir=None, **kwargs)
```
This method loads the sequence to graph model (aka parser).
If no `model_dir` is not supplied the default of `amrlib/data/model_stog` is used.

`kwargs` can be used to pass parameters such as `device`, `batch_size`, `beam_size`, etc to the
inference routine.

See specific [model descriptions](https://amrlib.readthedocs.io/en/latest/models/) for additional
parameters and their use.

The function returns a `STOGInferenceBase` type object which is a simple abstract base class for the underlying model.


**Inference.parse_sents()**
```
graphs = parse_sents(sents, add_metadata=True)
```
This method takes a list of sentence strings and converts them into a list of AMR graphs.
The optional parameter `add_metadata` tells the system if metadata such as "id", "snt", etc..
should appear at the top of the graph string.


### Example
```
import amrlib
stog = amrlib.load_stog_model()
graphs = stog.parse_sents(['This is a test of the system.', 'This is a second sentence.'])
for graph in graphs:
    print(graph)
```

## Graph to Sequence Functions (Generation)
**load_gtos_model()**
```
gtos = load_gtos_model(model_dir=None, **kwargs)
```
This method loads the graph to sequence model (aka generator).
If no `model_dir` is specified the default `amrlib/data/model_gtos` is used.

`kwargs` can be used to pass parameters such as `device`, `batch_size`, `num_beams`, `num_ret_seq`, etc.

See specific [model descriptions]((https://amrlib.readthedocs.io/en/latest/models/) for additional
parameters and their use.

`device` is automatically selected but you can pass `cpu` or `cuda:0` if needed.

The function returns a `GTOSInferenceBase` type object which is a simple abstract base class for the underlying model.

**Inference.generate()**
```
sents, clips = generate(graphs, disable_progress=False)
```
This method takes a list of AMR graph strings and returns a list of sentence strings and a list of booleans.
The boolean list `clips` tells if any of the returned sentences were clipped as a result of
the tokenized graph being too long for the model.

`disable_progress` can be used to turn off the default `tqdm` progress bar.

**Inference.get_ans_group()**
```
sents = get_ans_group(answers, group_num)
```
This is a simple slicing function that returns all the sentences associated with the input graph number.

`answers` is the returned list from `generate()` and `group_num` is the input graph number.  The method
will return `num_req_seq` sentence strings.


### Example
```
import amrlib
gtos = amrlib.load_gtos_model()
sents, _ = gtos.generate(graphs, disable_progress=True)
for sent in sents:
    print(sent)

for gnum in range(len(graphs)):
    print('graph number', gnum)
    for sent in gtos.get_ans_group(sents, gnum):
        print(sent)
```
