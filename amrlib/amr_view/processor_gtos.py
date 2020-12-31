import logging
from   threading import Thread
from   ..models.model_factory import load_inference_model
from   ..graph_processing.amr_loading import split_amr_meta

logger = logging.getLogger(__name__)


class ProcessorGTOS(object):
    def __init__(self, config, disabled=False):
        self.model_dir   = config['gtos_model_dir']
        self.num_ret_seq = config.get('gtos_num_ret_seq', 1)
        self.num_beams   = config.get('gtos_num_beams', 1)
        self.batch_size  = config.get('gtos_batch_size', 1)
        self.device      = config.get('gtos_device')
        self.inference   = None
        if disabled:
            logger.warning('!!! ProcessorGTOS disabled for debug !!!')
        else:
            lm_thread = Thread(target=self.load_model)  # loads self.inference
            lm_thread.start()

    def is_ready(self):
        return self.inference is not None

    def run(self, amr_text):
        if self.inference is None:
            return
        answers, clips = self.inference.generate([amr_text], disable_progress=True)
        if clips[0]:
            logger.warning('Graph was clipped')
        # Concatenate multiple return sequences
        string = ''
        for i, ans in enumerate(answers):
            string += '%2d)  %s\n' % (i+1, ans)
        return string[:-1]  # strip final line-feed

    def load_model(self):
        self.inference = load_inference_model(self.model_dir, num_beams=self.num_beams,
                            device=self.device, num_ret_seq=self.num_ret_seq,
                            batch_size=self.batch_size)
        logger.info('Graph to sequence model ready')
        print('Graph to sequence model ready')
