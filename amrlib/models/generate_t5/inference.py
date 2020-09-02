import logging
import torch
from   tqdm import tqdm
from   transformers import T5ForConditionalGeneration, T5Tokenizer

logger = logging.getLogger(__name__)


class Inference(object):
    def __init__(self, model_dir, **kwargs):
        default_device     = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        device             = kwargs.get('device', default_device)
        self.device        = torch.device(device)
        self.model         = T5ForConditionalGeneration.from_pretrained(model_dir).to(self.device)
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

    # generate sentences from a list of AMR text graphs
    # For generate params see https://huggingface.co/transformers/master/main_classes/model.html
    def generate(self, graphs, disable_progress=False):
        assert isinstance(graphs, list)
        sents = []
        clips = []
        dataloader = torch.utils.data.DataLoader(graphs, batch_size=self.batch_size)
        for batch in tqdm(dataloader, disable=disable_progress):
            # Form encodings and tokenize
            input_text = ['%s %s' % (graph, self.tokenizer.eos_token) for graph in batch]
            input_encodings = self.tokenizer.batch_encode_plus(input_text, pad_to_max_length=True,
                                                               truncation=True,
                                                               max_length=self.max_graph_len)
            input_ids      = torch.LongTensor(input_encodings['input_ids']).to(self.device)
            attention_mask = torch.LongTensor(input_encodings['attention_mask']).to(self.device)
            outs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                       max_length=self.max_sent_len, early_stopping=True,
                                       num_beams=self.num_beams,
                                       num_return_sequences=self.num_ret_seq)
            outs = [self.tokenizer.decode(ids) for ids in outs]
            sents.extend(outs)
            # Check if tokenized input_ids end with a pad or an eos token </s>.
            # If not, it was clipped
            clip = [ie[-1] not in self.seq_ends for ie in input_encodings['input_ids']]
            clips.extend(clip)
        return sents, clips

    # When num_ret_seq > 1, additional sentences are appended to the list, after the first
    # This is a simply extracts a group of them.  The length of the return is self.num_ret_seq
    def get_ans_group(self, answers, group_num):
        return answers[group_num*self.num_ret_seq:(group_num+1)*self.num_ret_seq]
