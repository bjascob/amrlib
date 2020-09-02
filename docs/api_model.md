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
stog = load_stog_model(model_dir=None, model_fn=None, **kwargs)
```
This method loads the sequence to graph model (aka parser).
If no `model_dir` or `model_fn` is not supplied the default of `amrlib/data/model_stog` and
`model.pt` are used, respectively.

`kwargs` can be used to pass parameters such as `device`, `batch_size`, `beam_size`, `alpha`
and `max_time_step` to the inference routine.

`device` is automatically selected but you can pass `cpu` or `cuda:0` if needed.

`beam_size` can be modified (default is 8) to increase or decrease performance.

See `amrlib/models/parse_gsii/inference.py` for implementation details.

The function returns a `parse_gsii.Inference` object.


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

`kwargs` can be used to pass parameters such as `device`, `batch_size`, `num_beams`, `num_ret_seq`.

`device` is automatically selected but you can pass `cpu` or `cuda:0` if needed.

`num_beams` (default is 1, aka greedy) can be increased as needed, to get a slightly higher BLUE
score. However, large values will likely require setting `batch_size` (default 32) lower when running
on the gpu.

`num_ret_seq` controls the number of returned sentences.  `num_beams` must be >= `num_ret_seq`.
Additional returned sentences will show up in order of score (high to low) on the returned list.
Note that a single list is returned, not a list of list. You can use `get_ans_group` (below) to
extract groupings if needed.

See `amrlib/models/generate_t5/inference.py` for implementation details.

The function returns a `generate_t5.Inference` object.

**Inference.generate()**
```
sents, clips = generate(graphs, disable_progress=False)
```
This method takes a list of AMR graph strings and returns a list of sentence strings and a list of booleans.
The boolean list `clips` tells if any of the returned sentences were clipped as a result of
the tokenized graph being too long for the model. Note that that model limit is 512 tokens
(not string characters) and roughly 2.5% of the LDC2020T02 graphs need to be clipped.

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
