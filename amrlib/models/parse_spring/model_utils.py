from   glob import glob
import torch
from   transformers import AutoConfig
from   transformers import BartForConditionalGeneration
from   .dataset import AMRDataset, AMRDatasetTokenBatcherAndLoader
from   .tokenization_bart import AMRBartTokenizer, PENMANBartTokenizer
from   .amr_rw import read_raw_amr_data


def instantiate_model_and_tokenizer(name=None, additional_tokens_smart_init=True, dropout=0.15,
        attention_dropout=0.15, collapse_name_ops=False, penman_linearization=False,
        use_pointer_tokens=False, raw_graph=False):
    if raw_graph:
        assert penman_linearization
    skip_relations = False
    if name is None:
        name = 'facebook/bart-large'
    if name == 'facebook/bart-base':
        tokenizer_name = 'facebook/bart-large'
    else:
        tokenizer_name = name
    config = AutoConfig.from_pretrained(name)
    config.output_past = False
    config.no_repeat_ngram_size = 0
    config.prefix = " "
    config.output_attentions = True
    config.dropout = dropout
    config.attention_dropout = attention_dropout
    # Setup the tokenizer
    if penman_linearization:
        tokenizer = PENMANBartTokenizer.from_pretrained(tokenizer_name, collapse_name_ops=collapse_name_ops,
            use_pointer_tokens=use_pointer_tokens, raw_graph=raw_graph, config=config)
    else:
        tokenizer = AMRBartTokenizer.from_pretrained(tokenizer_name, collapse_name_ops=collapse_name_ops,
            use_pointer_tokens=use_pointer_tokens, config=config)
    # Load the transformers model from the base model (ie.. facebook/bart-large).
    # if .from_pretrained(model, state_dict=x) is passed the model's state dict it will load those weights
    # instead of the base model. However, since the emeddings have been resize, the reload has to happen later
    # or there will be an error.
    model = BartForConditionalGeneration.from_pretrained(name, config=config)
    # Add the new AMR specific tokens to the model and modify the embeddings
    model.resize_token_embeddings(len(tokenizer.encoder))
    if additional_tokens_smart_init:
        modified = 0
        for tok, idx in tokenizer.encoder.items():
            tok = tok.lstrip(tokenizer.INIT)
            if idx < tokenizer.old_enc_size:
                continue
            elif tok.startswith('<pointer:') and tok.endswith('>'):
                tok_split = ['pointer', str(tok.split(':')[1].strip('>'))]
            elif tok.startswith('<'):
                continue
            elif tok.startswith(':'):
                if skip_relations:
                    continue
                elif tok.startswith(':op'):
                    tok_split = ['relation', 'operator', str(int(tok[3:]))]
                elif tok.startswith(':snt'):
                    tok_split = ['relation', 'sentence', str(int(tok[4:]))]
                elif tok.startswith(':ARG'):
                    tok_split = ['relation', 'argument', str(int(tok[4:]))]
                else:
                    tok_split = ['relation'] + tok.lstrip(':').split('-')
            else:
                tok_split = tok.split('-')
            tok_split_ = tok_split
            tok_split = []
            for s in tok_split_:
                s_ = s + tokenizer.INIT
                if s_ in tokenizer.encoder:
                    tok_split.append(s_)
                else:
                    tok_split.extend(tokenizer._tok_bpe(s))
            vecs = []
            for s in tok_split:
                idx_split = tokenizer.encoder.get(s, -1)
                if idx_split > -1:
                    vec_split = model.model.shared.weight.data[idx_split].clone()
                    vecs.append(vec_split)
            if vecs:
                vec = torch.stack(vecs, 0).mean(0)
                noise = torch.empty_like(vec)
                noise.uniform_(-0.1, +0.1)
                model.model.shared.weight.data[idx] = vec + noise
                modified += 1
    return model, tokenizer


# Load the model weights and optionally, the optimizer state, from the file
def load_state_dict_from_checkpoint(checkpoint_fn, model, optimizer=None, scheduler=None):
    model_dict = torch.load(checkpoint_fn, map_location='cpu')
    model.load_state_dict(model_dict['model'])
    if optimizer is not None and 'optimizer' in model_dict:
        optimizer.load_state_dict(model_dict['optimizer'])
    if scheduler is not None and 'scheduler' in model_dict:
        scheduler.load_state_dict(model_dict['scheduler'])


# Load all the AMR files in the glob_pattern into the dataloader (for training or evaluation)
def get_dataloader(tokenizer, glob_pattern, batch_size=500, evaluation=True, use_recategorization=False,
                        remove_longer_than=None, remove_wiki=False, dereify=True, device='cpu'):
    graphs  = read_raw_amr_data(glob_pattern, use_recategorization, remove_wiki=remove_wiki, dereify=dereify)
    dataset = AMRDataset(tokenizer, graphs, remove_longer_than=remove_longer_than)
    loader  = AMRDatasetTokenBatcherAndLoader(dataset, batch_size=batch_size, shuffle=not evaluation, device=device)
    return loader
