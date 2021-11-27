import logging
import torch
from   tqdm import tqdm
import penman
from   penman.models.noop import NoOpModel
import transformers
from   transformers import T5ForConditionalGeneration, T5Tokenizer
from   .penman_serializer import PenmanDeSerializer
from   ..inference_bases import STOGInferenceBase
from   ...graph_processing.amr_loading import split_amr_meta


logger = logging.getLogger(__name__)


class Inference(STOGInferenceBase):
    def __init__(self, model_dir=None, model_fn=None, model=None, tokenizer=None, config=None, **kwargs):
        default_device     = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device        = torch.device(kwargs.get('device', default_device))
        self.batch_size    = kwargs.get('batch_size', 12)
        self.num_beams     = kwargs.get('num_beams',   4)
        self.num_ret_seq   = self.num_beams
        self.ret_raw_gen   = kwargs.get('ret_raw_gen', False)   # Use only for debug
        # Load the model from file
        if model_dir is not None:
            model     = T5ForConditionalGeneration.from_pretrained(model_dir).to(self.device)
            tok_name  = kwargs.get('tok_name_or_path', 't5-base')
            tokenizer = T5Tokenizer.from_pretrained(tok_name)
            # model_parse_t5-v0_1_0 used the key "translation_amr_to_text" (copy error from generate_t5 code)
            config = model.config.task_specific_params.get('parse_amr')
            if config is None:
                config = model.config.task_specific_params.get('translation_amr_to_text')
        # Use the passed in values
        elif model is not None and tokenizer is not None and config is not None:
            pass
        else:
            raise ValueError('Either pass in the model directory or the model, tokenizer and config.')
        # Add to the class
        self.model         = model.to(self.device)
        self.tokenizer     = tokenizer
        self.max_sent_len  = config['max_in_len']
        self.max_graph_len = config['max_out_len']

    # Generate sentences from a list of sentence strings
    @torch.no_grad()
    def parse_sents(self, sents, add_metadata=True, disable_progress=True):
        assert isinstance(sents, list)
        # Sort by sentence length for faster batching
        # Put the longest first so that inference speeds up as it progresses, instead of slowing down.
        data  = [(s, i) for i, s in enumerate(sents)]
        data  = sorted(data, key=lambda x:len(x[0]), reverse=True)
        # Loop though batches
        clips = []
        graphs_generated = [None]*len(sents)*self.num_ret_seq
        self.model.eval()
        pbar = tqdm(total=len(sents), ncols=100, position=0, leave=True, disable=disable_progress)
        for batch in self._chunk(data, self.batch_size):
            input_text  = [x[0] for x in batch]
            sent_indxes = [x[1] for x in batch]
            # Form encodings and tokenize (padding=True => pad to the longest)
            input_encodings = self.tokenizer(input_text, padding=True, truncation=True,
                                max_length=self.max_sent_len, return_overflowing_tokens=True)
            # Check if any graphs were truncated (requires return_overflowing_tokens=True)
            clip = [l > 0 for l in input_encodings['num_truncated_tokens']]
            clips.extend(clip)
            # Convert to tensors
            input_ids      = torch.LongTensor(input_encodings['input_ids']).to(self.device)
            attention_mask = torch.LongTensor(input_encodings['attention_mask']).to(self.device)
            # Generate the batch ids and convert to back to tokens
            outs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                       max_length=self.max_graph_len, early_stopping=True,
                                       num_beams=self.num_beams, num_return_sequences=self.num_ret_seq)
            outs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
            # De-sort the output token data. There are self.num_ret_seq returned for each sentence
            for bidx in range(len(batch)):
                sidx = sent_indxes[bidx]
                graphs_generated[self._group_slice(sidx)] = outs[self._group_slice(bidx)]
            pbar.update(len(batch))
        pbar.close()
        # For debugging and sanity check
        if self.ret_raw_gen:
            return graphs_generated, clips
        assert not any(g is None for g in graphs_generated)
        # Get the top result that properly deserializes. graphs_generated is len(sents)*num_ret_seq
        graphs_final = [None]*len(sents)
        for snum in range(len(sents)):
            if clips[snum]:
                logger.error('Sentence number %d was clipped for length' % snum)
            raw_graphs = graphs_generated[self._group_slice(snum)]
            for bnum, g in enumerate(raw_graphs):
                gstring = PenmanDeSerializer(g).get_graph_string()
                if gstring is not None:
                    graphs_final[snum] = gstring
                    break   # stop deserializing candidates when we find a good one
                else:
                    logger.error('Failed to deserialize, snum=%d, beam=%d' % (snum, bnum))
        # Add metadata
        if add_metadata:
            graphs_final = ['# ::snt %s\n%s' % (s, g) if g is not None else None for s, g in zip(sents, graphs_final)]
        return graphs_final

    # Return a slice operator to extract the models ouput group based on the input index
    # The model returns self.num_ret_seq * length(input) as a flat list.
    def _group_slice(self, input_idx):
        return slice(input_idx * self.num_ret_seq, (input_idx + 1) * self.num_ret_seq)

    # Yield successive n-sized chunks from lst.
    @staticmethod
    def _chunk(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    # parse a list of spacy spans (ie.. span has list of tokens)
    def parse_spans(self, spans, add_metadata=True):
        sents = [s.text.strip() for s in spans]
        graphs = self.parse_sents(sents, add_metadata, disable_progress=True)
        return graphs
