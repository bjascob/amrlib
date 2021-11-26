import os
import json
import logging
import torch
from   tqdm import tqdm
import penman
from   penman.models.amr import model as amr_model
from   ..inference_bases import STOGInferenceBase
from   .model_utils import instantiate_model_and_tokenizer, load_state_dict_from_checkpoint
from   .postprocessing import ParsedStatus

logger = logging.getLogger(__name__)


class Inference(STOGInferenceBase):
    invalid_graph = penman.decode('()')
    def __init__(self, model_dir=None, model_fn=None, model=None, tokenizer=None, config=None, **kwargs):
        logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)   # skip tokenizer warning
        default_device     = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device        = torch.device(kwargs.get('device', default_device))
        self.batch_size    = kwargs.get('batch_size', 8)    # in number of sentences
        self.num_beams     = kwargs.get('num_beams',  5)
        # Load the model from file
        if model_dir is not None and model_fn is not None:
            config, model, tokenizer = self.load_model(model_dir, model_fn)
        elif model is not None and tokenizer is not None and config is not None:
            pass
        else:
            raise ValueError('Either enter a model name and directory or pass in the model, tokenizer and config')
        self.config             = config
        self.model              = model.to(self.device)
        self.tokenizer          = tokenizer
        self.restore_name_ops   = self.config['collapse_name_ops']

    # Load the model from a file
    @staticmethod
    def load_model(model_dir, model_fn):
        with open(os.path.join(model_dir, 'config.json')) as f:
            config = json.load(f)
        model_name = config['model']
        model, tokenizer = instantiate_model_and_tokenizer(model_name, dropout=0., attention_dropout=0.,
                penman_linearization=config['penman_linearization'],
                use_pointer_tokens=config['use_pointer_tokens'], raw_graph=config['raw_graph'])
        load_state_dict_from_checkpoint(os.path.join(model_dir, model_fn), model)
        return config, model, tokenizer

    # Generate sentences from a list of sentence strings
    def parse_sents(self, sents, add_metadata=True, return_penman=False, disable_progress=True, pbar_desc=None):
        assert isinstance(sents, list)
        # Loop though batches
        gen_graphs = []
        dataloader = torch.utils.data.DataLoader(sents, batch_size=self.batch_size, shuffle=False)
        pbar = tqdm(total=len(dataloader.dataset), disable=disable_progress, ncols=100, desc=pbar_desc)
        for batch in dataloader:
            # I'm Ignoring potential for clipped sentences.  If return_overflowing_tokens=True is passed into
            # the the lower level call to batch_encode_plus(), I could get these back out if needed.
            # Bart supports up to 1024 input tokens so this would have to be paragraphs of text, all
            # concatenated into a single "sent", in order to overflow.  This isn't valid for AMR anyway.
            x, _ = self.tokenizer.batch_encode_sentences(batch, device=self.device)
            # model.config.max_length=20 is the base model. Set this much higher for generating AMR graphs.
            with torch.no_grad():
                model_out = self.model.generate(**x, max_length=512, num_beams=self.num_beams)
            # re-encode the model output
            assert len(model_out) == len(batch)
            for tokk, sent in zip(model_out, batch):
                graph, status, _ = self.tokenizer.decode_amr(tokk.tolist(), restore_name_ops=self.restore_name_ops)
                # Handle status errors (also has ParsedStatus.FIXED for fixed disconnected graphs)
                if status == ParsedStatus.BACKOFF:
                    graph = self.invalid_graph
                # In Penman 1.2.0, metadata does not impact penam.Graph.__eq__() so code checking for
                # Inference.invalid_graph should still work, even if 'snt' metadata is different.
                if add_metadata:
                    graph.metadata['snt'] = sent
                gen_graphs.append(graph)
            pbar.update(len(batch))
        pbar.close()
        # Return the penman graphs
        if return_penman:
            return gen_graphs
        # The required behavior across all parse_mdoels, is to return graphs as strings by default
        gstrings = [penman.encode(g, indent=6, model=amr_model) for g in gen_graphs]
        return gstrings

    # parse a list of spacy spans (ie.. span has list of tokens)
    def parse_spans(self, spans, add_metadata=True):
        sents = [s.text.strip() for s in spans]
        graphs = self.parse_sents(sents, add_metadata, disable_progress=True)
        return graphs
