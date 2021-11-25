import os
import logging
from   . import defaults
from   .utils.downloader import download_model, set_symlink
from   .models.model_factory import load_inference_model

logger = logging.getLogger(__name__)


# Library version number
__version__ = '0.6.0'


# Download the model to and un-tar it in the mdata_dir
# ie.. download('model_generate_t5', 'http://127.0.0.1:8000/model_generate_t5-v_0_0_0.tar.gz')
# the model_name will the the linkname in the mdata_dir pointing to the new model's directory
# For testing with local data, cd to the directory with the models run a local file server with...
# `python3 -m http.server 8000`
def download(model_name, url, mdata_dir=None, rm_tar=True, set_links=True):
    mdata_dir = mdata_dir if mdata_dir is not None else defaults.data_dir
    model_dir = download_model(url, mdata_dir, rm_tar)
    if set_links and model_dir is not None:
        src = os.path.basename(model_dir)           # strip the path
        dst = os.path.join(mdata_dir, model_name)   # add the path
        set_symlink(src, dst)


#### Model Loading ####

# Load the sentence to graph model
stog_model = None   # cache model once loaded
def load_stog_model(model_dir=None, **kwargs):
    model_dir = model_dir if model_dir is not None else os.path.join(defaults.data_dir, 'model_stog')
    global stog_model
    stog_model = load_inference_model(model_dir, **kwargs)
    return stog_model

# Load the graph to sentence model
gtos_model = None   # cache model once loaded
def load_gtos_model(model_dir=None, **kwargs):
    global gtos_model
    model_dir = model_dir if model_dir is not None else os.path.join(defaults.data_dir, 'model_gtos')
    gtos_model = load_inference_model(model_dir, **kwargs)
    return gtos_model


#### Spacy extensions ####

# Extension for spacy Docs
def spacy_stog_doc(doc):
    spans = [span for span in doc.sents]
    global stog_model
    if stog_model is None:
        load_stog_model()
    return stog_model.parse_spans(spans)

# Extension for spacy Spans (ie.. sentences)
def spacy_stog_span(span):
    global stog_model
    if stog_model is None:
        load_stog_model()
    return stog_model.parse_spans([span])

# Hook in the system as a spacy extension
def setup_spacy_extension():
    import spacy
    spacy.tokens.doc.Doc.set_extension(  'to_amr', method=spacy_stog_doc,  force=True)
    spacy.tokens.span.Span.set_extension('to_amr', method=spacy_stog_span, force=True)
