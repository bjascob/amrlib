from   glob import glob
import penman
from   penman.model import Model
from   penman.models.noop import model as noop_model


# Load a one or more AMR files. (ie... read_raw_amr_data('xyz/*.txt',...) and return penman graphs
def read_raw_amr_data(glob_pattern, use_recategorization=False, dereify=True, remove_wiki=False):
    graphs = []
    for path in sorted(glob(glob_pattern)):
        graphs.extend(load_amr_file(path, dereify=dereify, remove_wiki=remove_wiki))
    assert graphs
    if use_recategorization:
        for g in graphs:
            metadata = g.metadata
            metadata['snt_orig'] = metadata['snt']
            tokens = eval(metadata['tokens'])
            metadata['snt'] = ' '.join([t for t in tokens if not ((t.startswith('-L') or t.startswith('-R')) and t.endswith('-'))])
    return graphs


# Load the AMR text file using dereify (aka normalization, ie.. invert arg0-of) modify the :wiki tags
# remove_wiki = "replace" to change them to :wiki +   or "remove" to remove the triple completely.
def load_amr_file(source, dereify=None, remove_wiki=False):
    assert remove_wiki in (False, 'replace', 'remove')
    # Select the model to use
    if dereify is None or dereify:  # None or True (odd way to do default logic)
        model = Model()             # default penman model, same as load(..., model=None)
    else:                           # False
        model = noop_model
    # Load the data
    out = penman.load(source=source, model=model)
    # Remove or replace the wiki tags
    if remove_wiki == 'remove':
        for i in range(len(out)):
            out[i] = _remove_wiki(out[i])
    elif remove_wiki == 'replace':
        for i in range(len(out)):
            out[i] = _replace_wiki(out[i])
    return out

# Replace all :wiki triples with :wiki +
def _replace_wiki(graph):
    metadata = graph.metadata
    triples = []
    for t in graph.triples:
        v1, rel, v2 = t
        if rel == ':wiki':
            t = penman.Triple(v1, rel, '+')
        triples.append(t)
    graph = penman.Graph(triples)
    graph.metadata = metadata
    return graph

# Completely remove all :wiki triples
def _remove_wiki(graph):
    metadata = graph.metadata
    triples = []
    for t in graph.triples:
        v1, rel, v2 = t
        if rel == ':wiki':
            continue
        triples.append(t)
    graph = penman.Graph(triples)
    graph.metadata = metadata
    return graph
