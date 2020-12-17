import warnings
warnings.simplefilter('ignore')
import logging
from   threading import Thread
from   ..models.model_factory import load_inference_model
from   ..graph_processing.annotator import load_spacy

logger = logging.getLogger(__name__)

# Setence to Graph processor
class ProcessorSTOG(object):
    def __init__(self, config, disabled=False):
        self.model_dir      = config['stog_model_dir']
        self.model_fn       = config['stog_model_fn']
        self.device         = config['stog_device']
        self.show_metadata  = True
        self.inference      = None
        if disabled:
            logger.warning('!!! ProcessorSTOG disabled for debug !!!')
        else:
            lm_thread = Thread(target=self.load_model)  # loads self.inference
            lm_thread.start()

    def is_ready(self):
        return self.inference is not None

    def run(self, sent):
        if self.inference is None:
            return
        entries = self.inference.parse_sents([sent], self.show_metadata)
        return entries[0]

    def load_model(self):
        load_spacy()    # pre-load this for the annotator
        self.inference = load_inference_model(self.model_dir, device=self.device)
        logger.info('Sequence to graph model ready')
        print('Sequence to graph model ready')
