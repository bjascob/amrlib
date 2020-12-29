import logging
import torch
from   tqdm import tqdm
from   transformers import T5ForConditionalGeneration, T5Tokenizer
from   ..inference_bases import GTOSInferenceBase
from   ...graph_processing.amr_loading import split_amr_meta

logger = logging.getLogger(__name__)


class Inference(GTOSInferenceBase):
    def __init__(self, model_dir, model_fn=None, **kwargs):
        default_device     = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        device             = kwargs.get('device', default_device)
        self.device        = torch.device(device)
        # The following produces a logger warning that we can ignore so eliminate temporarily set the level higher
        xfm_logger         = logging.getLogger('transformers.modeling_utils')
        original_level     = xfm_logger.getEffectiveLevel()
        xfm_logger.setLevel(logging.ERROR)
        self.model         = T5ForConditionalGeneration.from_pretrained(model_dir).to(self.device)
        xfm_logger.setLevel(original_level)
        # End logger ignore warning
        self.max_graph_len = self.model.config.task_specific_params['translation_amr_to_text']['max_in_len']
        self.max_sent_len  = self.model.config.task_specific_params['translation_amr_to_text']['max_out_len']
        tokenizer_name     = kwargs.get('tokenizer_name', 't5-base')    # name or path
        self.tokenizer     = T5Tokenizer.from_pretrained(tokenizer_name)
        self.seq_ends      = set([self.tokenizer.eos_token_id, self.tokenizer.pad_token_id])
        self.batch_size    = kwargs.get('batch_size', 32)
        self.num_beams     = kwargs.get('num_beams',   1)  # 1 => greedy
        self.num_ret_seq   = kwargs.get('num_ret_seq', 1)
        if self.num_ret_seq > self.num_beams:
            logger.warn('Need at least as many beams as returned sequences - increasing beam count')
            self.num_beams = self.num_ret_seq

    # Generate sentences from a list of AMR text graphs
    # For generate params see https://huggingface.co/transformers/master/main_classes/model.html
    def generate(self, graphs, disable_progress=True):
        assert isinstance(graphs, list)
        # Make sure the user isn't passing in meta-data, only the graph strings
        stripped_graphs = []
        for graph in graphs:
            meta_lines, graph_lines = split_amr_meta(graph)
            stripped_graphs.append(' '.join(graph_lines))
        # Loop though batches
        sents = []
        clips = []
        dataloader = torch.utils.data.DataLoader(stripped_graphs, batch_size=self.batch_size)
        for batch in tqdm(dataloader, disable=disable_progress):
            # Form encodings and tokenize
            input_text = ['%s' % graph for graph in batch]
            input_encodings = self.tokenizer.batch_encode_plus(input_text, padding=True,
                                    truncation=True, max_length=self.max_graph_len,
                                    return_overflowing_tokens=True)
            # Check if any graphs were truncated (requires return_overflowing_tokens=True)
            clip = [l > 0 for l in input_encodings['num_truncated_tokens']]
            clips.extend(clip)
            # Convert to tensors
            input_ids      = torch.LongTensor(input_encodings['input_ids']).to(self.device)
            attention_mask = torch.LongTensor(input_encodings['attention_mask']).to(self.device)
            # Generate
            outs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                        max_length=self.max_sent_len, early_stopping=True, num_beams=self.num_beams,
                        num_return_sequences=self.num_ret_seq)
            outs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
            sents.extend(outs)
        return sents, clips

    # When num_ret_seq > 1, additional sentences are appended to the list, after the first
    # This is a simply extracts a group of them.  The length of the return is self.num_ret_seq
    def get_ans_group(self, answers, group_num):
        return answers[group_num*self.num_ret_seq:(group_num+1)*self.num_ret_seq]
