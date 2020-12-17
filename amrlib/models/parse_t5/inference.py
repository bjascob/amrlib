import logging
import torch
from   tqdm import tqdm
import penman
from   penman.models.noop import NoOpModel
from   transformers import T5ForConditionalGeneration, T5Tokenizer
from   .penman_serializer import PenmanDeSerializer
from   ..inference_bases import STOGInferenceBase
from   ...graph_processing.amr_loading import split_amr_meta


logger = logging.getLogger(__name__)


class Inference(STOGInferenceBase):
    def __init__(self, model_dir, model_fn=None, **kwargs):
        default_device     = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        device             = kwargs.get('device', default_device)
        self.device        = torch.device(device)
        self.model         = T5ForConditionalGeneration.from_pretrained(model_dir).to(self.device)
        self.max_sent_len  = self.model.config.task_specific_params['translation_amr_to_text']['max_in_len']
        self.max_graph_len = self.model.config.task_specific_params['translation_amr_to_text']['max_out_len']
        tokenizer_name     = kwargs.get('tokenizer_name', 't5-base')    # name or path
        self.tokenizer     = T5Tokenizer.from_pretrained(tokenizer_name)
        self.seq_ends      = set([self.tokenizer.eos_token_id, self.tokenizer.pad_token_id])
        self.batch_size    = kwargs.get('batch_size', 12)
        self.num_beams     = kwargs.get('num_beams',   4)       # 1 => greedy
        self.num_ret_seq   = self.num_beams
        self.ret_raw_gen   = kwargs.get('ret_raw_gen', False)   # Use only for debug

    # Generate sentences from a list of sentence strings
    # For generate params see https://huggingface.co/transformers/master/main_classes/model.html
    def parse_sents(self, sents, add_metadata=True, disable_progress=True):
        assert isinstance(sents, list)
        # Loop though batches
        graphs_generated = []
        clips  = []
        dataloader = torch.utils.data.DataLoader(sents, batch_size=self.batch_size)
        for batch in tqdm(dataloader, disable=disable_progress):
            # Form encodings and tokenize
            input_text = ['%s %s' % (sent, self.tokenizer.eos_token) for sent in batch]
            input_encodings = self.tokenizer.batch_encode_plus(input_text, padding=True,
                                                               truncation=True,
                                                               max_length=self.max_sent_len)
            input_ids      = torch.LongTensor(input_encodings['input_ids']).to(self.device)
            attention_mask = torch.LongTensor(input_encodings['attention_mask']).to(self.device)
            outs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                       max_length=self.max_graph_len, early_stopping=True,
                                       num_beams=self.num_beams, num_return_sequences=self.num_ret_seq)
            outs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
            graphs_generated.extend(outs)
            # Check if tokenized input_ids end with a pad or an eos token </s>.
            # If not, it was clipped
            clip = [ie[-1] not in self.seq_ends for ie in input_encodings['input_ids']]
            clips.extend(clip)
        # For debugging only ...
        # Note: in this mode we're returning 2 lists of num_ret_seq * len(sents) instead of
        # one list of len(sents) as in the default run-time mode
        if self.ret_raw_gen:
            return graphs_generated, clips
        # Extract the top result that isn't clipped and will deserialize
        # At this point "graphs_generated" and "clips" have num_ret_seq for each sent * len(sents)
        graphs_final = [None]*len(sents)
        for snum in range(len(sents)):
            if clips[snum]:
                logger.warning('Sentence number %d was clipped for length' % snum)
            raw_graphs = graphs_generated[snum*self.num_ret_seq:(snum+1)*self.num_ret_seq]
            for bnum, g in enumerate(raw_graphs):
                gstring = PenmanDeSerializer(g).get_graph_string()
                if gstring is not None:
                    graphs_final[snum] = gstring
                    break   # stop deserializing candidates when we find a good one
                else:
                    logger.warning('Failed to deserialize, snum=%d, beam=%d' % (snum, bnum))
        # Add metadata
        if add_metadata:
            graphs_final = ['# ::snt %s\n%s' % (s, g) for s, g in zip(sents, graphs_final) if g is not None]
        return graphs_final

    # parse a list of spacy spans (ie.. span has list of tokens)
    def parse_spans(self, spans, add_metadata=True):
        sents = [s.text.strip() for s in spans]
        graphs = self.parse_sents(sents, add_metadata, disable_progress=True)
        return graphs
