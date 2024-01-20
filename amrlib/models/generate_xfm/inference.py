import logging
import torch
from   tqdm import tqdm
from   transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from   .model_input_helper import ModelInputHelper
from   ..inference_bases import GTOSInferenceBase
from   ...graph_processing.amr_loading import split_amr_meta

logger = logging.getLogger(__name__)


class Inference(GTOSInferenceBase):
    def __init__(self, model_dir=None, model_fn=None, model=None, tokenizer=None, **kwargs):
        # Load params from passed in values
        self.batch_size    = kwargs.get('batch_size', 32)
        self.num_beams     = kwargs.get('num_beams',   1)  # 1 => greedy
        self.num_ret_seq   = kwargs.get('num_ret_seq', 1)
        if self.num_ret_seq > self.num_beams:
            logger.warn('Need at least as many beams as returned sequences - increasing beam count')
            self.num_beams = self.num_ret_seq
        # Load the model and tokenizer
        if model_dir is not None:
            default_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            device = kwargs.get('device', default_device)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
            config = model.config.task_specific_params['translation_amr_to_text']
            # Load the tokenizer. If kwargs passes in tok_name_or_path use that otherwise get from the config
            tokenizer_name = kwargs.get('tok_name_or_path', config['model_name_or_path'])
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=config['max_in_len'])
        # Use the passed in values
        elif model is not None and tokenizer is not None:
            config = model.config.task_specific_params['translation_amr_to_text']
        else:
            raise ValueError('Either pass in the model directory or the model and tokenizer.')
        # Assign other values
        self.device        = model.device
        self.model         = model
        self.tokenizer     = tokenizer
        self.max_graph_len = config['max_in_len']
        self.max_sent_len  = config['max_out_len']

    # Generate sentences from a list of AMR text graphs
    # If use_tense=True, then Penn-TreeBank tags will be added to the graph
    # If `pos_tag` in not in the metadata, spacy will be called to create them using the `snt` field
    # reannotate=True forces parsing of `snt` and recreation of the pos_tag data
    @torch.no_grad()
    def generate(self, graphs, disable_progress=True, use_tense=False, reannotate=False):
        assert isinstance(graphs, list)
        # Convert the incoming graphs to the format used for model input
        stripped_graphs = []
        for graph in graphs:
            # If adding tense information, try to to tag the graph, which requires the sentence
            # or annotations and then goes through an alignment.  If something goes wrong, log an
            # error and fallback to just using a graph converted to a string.
            if use_tense or reannotate:
                try:
                    gstring = ModelInputHelper(graph, reannotate=reannotate).get_tagged_oneline()
                except:
                    logger.error('Unable to add tense information to graph')
                    gstring = ModelInputHelper.gstring_to_oneline(graph)
            # If not adding tense info, just strip any metadata and convert to a single line
            else:
                gstring = ModelInputHelper.gstring_to_oneline(graph)
            stripped_graphs.append(gstring)
        # Loop though batches
        sents, clips = [], []
        dataloader = torch.utils.data.DataLoader(stripped_graphs, batch_size=self.batch_size)
        for batch in tqdm(dataloader, ncols=100, position=0, leave=True, disable=disable_progress):
            # Form encodings and tokenize
            input_text = ['%s' % graph for graph in batch]
            input_encodings = self.tokenizer(input_text, padding=True, truncation=True, max_length=self.max_graph_len)
            # Check if any graphs are at max_graph_len meaning they were probably truncated
            for bidx, iids in enumerate(input_encodings['input_ids']):
                clip_tf = (len(iids) == self.max_graph_len) and (iids[-1] != self.tokenizer.pad_token_id)
                clips.append(clip_tf)
            # Convert to tensors
            input_ids      = torch.LongTensor(input_encodings['input_ids']).to(self.device)
            attention_mask = torch.LongTensor(input_encodings['attention_mask']).to(self.device)
            # Generate
            early_stopping = True if self.num_beams > 1 else False
            outs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                        max_length=self.max_sent_len, early_stopping=early_stopping,
                        num_beams=self.num_beams, num_return_sequences=self.num_ret_seq)
            outs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
            sents.extend(outs)
        return sents, clips

    # When num_ret_seq > 1, additional sentences are appended to the list, after the first
    # This is a simply extracts a group of them.  The length of the return is self.num_ret_seq
    def get_ans_group(self, answers, group_num):
        return answers[group_num*self.num_ret_seq:(group_num+1)*self.num_ret_seq]
