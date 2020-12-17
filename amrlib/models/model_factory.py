import os
import errno
import json
import logging
import importlib
from   ..utils.md5sum import md5sum     

logger = logging.getLogger(__name__)


# Dymaically import the module / class and return the class definition
def dynamic_load(module_name, class_name, package='amrlib.models'): 
    module = importlib.import_module(module_name, package=package)
    my_class = getattr(module, class_name)
    return my_class


# Hard-coded logic to detect and return metadata for the initial 2 models released that
# weren't setup to work this way.
MD5SUM_CHUNKSIZE=2**20
def get_non_config_model(model_directory):
    # Check for model_parse_gsii-v0_1_0
    fpath = os.path.join(model_directory, 'model.pt')
    if os.path.exists(fpath):
        hash_id = md5sum(fpath, chunksize=MD5SUM_CHUNKSIZE, first_chunk_only=True)
        if hash_id == '09048d8ba12dc729d815963348d6e901':
            return {"version":"0.0.1", "model_type":"stog","inference_module":".parse_gsii.inference",
                    "inference_class":"Inference","model_fn":"model.pt"}
    # Check for model_generate_t5-v0_1_0
    fpath = os.path.join(model_directory, 'pytorch_model.bin')
    if os.path.exists(fpath):
        hash_id = md5sum(fpath, chunksize=MD5SUM_CHUNKSIZE, first_chunk_only=True)
        if hash_id == '786e3f9d33a6981ffae7c5f42a935cc9':
            return {"version":"0.0.1", "model_type":"gtos","inference_module":".generate_t5.inference",
                    "inference_class":"Inference","model_fn":"pytorch_model.bin"}
    # Doesn't match anything here
    return None


# Load the model in the model_directory, and override any arguments with kwargs
# First try to load the amrlib_meta.json file.  If one doesn't exist, try doing a partial md5sum
# to the known models.
def load_inference_model(model_directory, **kwargs):
    # Error check
    if not os.path.isdir(model_directory):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), model_directory)
    meta = None
    # See if an amrlib_meta.json file exist in the model directory, and if so load it
    fpath = os.path.join(model_directory, 'amrlib_meta.json')
    if os.path.exists(fpath):
        with open(fpath) as f:
            meta = json.load(f)
    # If we can't get it from a file, try the hard-coded values (used for initial models)
    else:
        logger.info('No amrlib_meta.json file, trying hard-coded config match')
        meta = get_non_config_model(model_directory)
    # Raise an error if we can't load at this point
    if not meta:
        msg = 'No meta-data (amrlib_meta.json or hard-coded) available for'
        raise FileNotFoundError(errno.ENOENT, msg, model_directory)
    # With the meta-data, load the model and instantiate it
    model_class = dynamic_load(module_name=meta['inference_module'], class_name=meta['inference_class'])
    model_kwargs = meta.get('kwargs', {})   # get any model kwargs from the meta-data
    model_kwargs.update(kwargs)             # override them with amything passed in
    model = model_class(model_directory, meta['model_fn'], **model_kwargs)
    return model
