import os
import json
import logging
import multiprocessing
from   tqdm import tqdm
import penman
from   penman.models.noop import NoOpModel
from   .amr_loading import load_amr_entries
from   .. import defaults

logger = logging.getLogger(__name__)


# Default set of tags to keep when annotatating the AMR.  Throw all others away
# To keep all, redefine this to None
keep_tags=set(['id','snt'])


# Annotate a file with multiple AMR entries and save it to the specified location
def annotate_file(indir, infn, outdir, outfn):
    load_spacy()
    graphs = []
    inpath = os.path.join(indir, infn)
    entries = load_amr_entries(inpath)
    pool = multiprocessing.Pool()
    #for pen in tqdm(map(_process_entry, entries), total=len(entries)):
    for pen in tqdm(pool.imap(_process_entry, entries), total=len(entries)):
        graphs.append(pen)
    pool.close()
    pool.join()
    infn = infn[:-3] if infn.endswith('.gz') else infn  # strip .gz if needed
    outpath = os.path.join(outdir, outfn)
    print('Saving file to ', outpath)
    penman.dump(graphs, outpath, indent=6)


# Annotate a single AMR string and return a penman graph
def annotate_graph(entry, tokens=None):
    load_spacy()
    return _process_entry(entry, tokens)

# Annotate a single penman AMR Graph and return a penman graph
def annotate_penman(pgraph, tokens=None):
    load_spacy()
    return _process_penman(pgraph, tokens)

# Worker process that takes in an amr string and returns a penman graph object
# Annotate the raw AMR entries with SpaCy to add the required ::tokens and ::lemmas fields
# plus a few other fields for future pre/postprocessing work that may be needed.
# Keep only tags in "keep_tags"
def _process_entry(entry, tokens=None):
    pen = penman.decode(entry)      # standard de-inverting penman loading process
    return _process_penman(pen, tokens)

# Split out the _process_entry for instances where the string is already converted to a penman graph
def _process_penman(pen, tokens=None):
    # Filter out old tags and add the tags from SpaCy parse
    global keep_tags
    if keep_tags is not None:
        pen.metadata = {k:v for k,v in pen.metadata.items() if k in keep_tags}  # filter extra tags
    # If tokens aren't supplied then annoate the graph
    if not tokens:
        global spacy_nlp
        assert spacy_nlp is not None
        tokens = spacy_nlp(pen.metadata['snt'])
    pen.metadata['tokens']   = json.dumps([t.text      for t in tokens])
    ner_tags = [t.ent_type_ if t.ent_type_ else 'O' for t in tokens]    # replace empty with 'O'
    pen.metadata['ner_tags'] = json.dumps(ner_tags)
    pen.metadata['ner_iob']  = json.dumps([t.ent_iob_  for t in tokens])
    pen.metadata['pos_tags'] = json.dumps([t.tag_      for t in tokens])
    # Create lemmas
    # The spaCy 2.0 lemmatizer returns -PRON- for pronouns so strip these (spaCy 3.x does not do this)
    # Don't try to lemmatize any named-entities or proper nouns.  Lower-case any other words.
    lemmas = []
    for t in tokens:
        if t.lemma_ == '-PRON-':    # spaCy 2.x only
            lemma = t.text.lower()
        elif t.tag_.startswith('NNP') or t.ent_type_ not in ('', 'O'):
            lemma = t.text
        else:
            lemma = t.lemma_.lower()
        lemmas.append(lemma)
    pen.metadata['lemmas'] = json.dumps(lemmas)
    return pen


# Take a graph string entry and process it through spacy to create metadata fields for
# tokens and lemmas using the snt_key.  If a 'verify_tok_key' is provided, compare the
# tokenization length for spacy tokenization to the space-tokenized ":tok" field and
# only return a graph is they match.
# This was added speficially for alignments but with a little work, could be harmonized with above
def add_lemmas(entry, snt_key, verify_tok_key=None):
    global spacy_nlp
    load_spacy()
    graph  = penman.decode(entry, model=NoOpModel())    # do not de-invert graphs
    doc        = spacy_nlp(graph.metadata[snt_key])
    nlp_tokens = [t.text for t in doc]
    graph.metadata['tokens'] = json.dumps(nlp_tokens)
    # Create lemmas
    # SpaCy's lemmatizer returns -PRON- for pronouns so strip these
    # Don't try to lemmatize any named-entities or proper nouns.  Lower-case any other words.
    lemmas = []
    for t in doc:
        if t.lemma_ == '-PRON-':
            lemma = t.text.lower()
        elif t.tag_.startswith('NNP') or t.ent_type_ not in ('', 'O'):
            lemma = t.text
        else:
            lemma = t.lemma_.lower()
        lemmas.append(lemma)
    graph.metadata['lemmas'] = json.dumps(lemmas)
    # If verify_tok_key is not None, verify that the new tokenization is the same as the existing
    # and only return the graph if the tokenized length is the same
    if verify_tok_key is not None:
        isi_tokens = graph.metadata[verify_tok_key].split()
        if len(isi_tokens) == len(lemmas) == len(nlp_tokens):
            return graph
        else:
            return None
    else:
        return graph


# Spacy NLP - lazy loader
# This will only load the model onece, even if called again with a different model name.
# Note that when multiprocessing, call this once from the main process (before using pool)
# to load it into the main processes, then when pool forks, it will be copied, otherwise it
# will be loaded multiple times.
spacy_nlp = None
def load_spacy(model_name=None):
    global spacy_nlp
    if spacy_nlp is not None:   # will return if a thread is already loading
        return
    model_name = model_name if model_name is not None else defaults.spacy_model_name
    #print('Loading SpaCy NLP Model:', model_name)
    import spacy
    spacy_nlp = spacy.load(model_name)
