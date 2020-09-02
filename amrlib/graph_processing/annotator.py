import os
import json
import logging
import multiprocessing
from   tqdm import tqdm
import penman
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


# Worker process that takes in an amr string and returns a penman graph object
# Annotate the raw AMR entries with SpaCy to add the required ::tokens and ::lemmas fields
# plus a few other fields for future pre/postprocessing work that may be needed.
# Keep only tags in "keep_tags"
def _process_entry(entry, tokens=None):
    pen = penman.decode(entry)
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
    # SpaCy's lemmatizer returns -PRON- for pronouns so strip these
    # Don't try to lemmatize any named-entities or proper nouns.  Lower-case any other words.
    lemmas = []
    for t in tokens:
        if t.lemma_ == '-PRON-':
            lemma = t.text.lower()
        elif t.tag_.startswith('NNP') or t.ent_type_ not in ('', 'O'):
            lemma = t.text
        else:
            lemma = t.lemma_.lower()
        lemmas.append(lemma)
    pen.metadata['lemmas'] = json.dumps(lemmas)
    return pen


# Spacy NLP - lazy loader
# This will only load the model onece, even if called again with a different model name.
spacy_nlp = None
def load_spacy(model_name=None):
    global spacy_nlp
    if spacy_nlp is not None:   # will return if a thread is already loading
        return
    model_name = model_name if model_name is not None else defaults.spacy_model_name
    print('Loading SpaCy NLP Model:', model_name)
    import spacy
    spacy_nlp = spacy.load(model_name)
