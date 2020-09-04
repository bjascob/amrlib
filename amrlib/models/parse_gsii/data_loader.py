import random
import torch
from   torch import nn
import numpy as np
from   .amr_graph import read_file
from   .vocabs import PAD, UNK, DUM, NIL, END, CLS


# Returns cp_seq as a list of lemma + '_' and mp_seq is a list of the lemmas
# plus the dictionaries to convert the tokens to an index
# This represents the potential concepts / attributes
def get_concepts(lem, vocab):
    cp_seq, mp_seq = [], []
    new_tokens = set()
    for le in lem:
        cp_seq.append(le + '_')
        mp_seq.append(le)
    for cp, mp in zip(cp_seq, mp_seq):
        if vocab.token2idx(cp) == vocab.unk_idx:
            new_tokens.add(cp)
        if vocab.token2idx(mp) == vocab.unk_idx:
            new_tokens.add(mp)
    nxt = vocab.size
    token2idx, idx2token = dict(), dict()
    for x in new_tokens:
        token2idx[x] = nxt
        idx2token[nxt] = x
        nxt += 1
    return cp_seq, mp_seq, token2idx, idx2token


def ListsToTensor(xs, vocab=None, local_vocabs=None, unk_rate=0.):
    pad = vocab.padding_idx if vocab else 0
    def toIdx(w, i):
        if vocab is None:
            return w
        if isinstance(w, list):
            return [toIdx(_, i) for _ in w]
        if random.random() < unk_rate:
            return vocab.unk_idx
        if local_vocabs is not None:
            local_vocab = local_vocabs[i]
            if (local_vocab is not None) and (w in local_vocab):
                return local_vocab[w]
        return vocab.token2idx(w)

    max_len = max(len(x) for x in xs)
    ys = []
    for i, x in enumerate(xs):
        y = toIdx(x, i) + [pad]*(max_len-len(x))
        ys.append(y)
    data = np.transpose(np.array(ys, dtype=np.int64))
    return data


def ListsofStringToTensor(xs, vocab, max_string_len=20):
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs:
        y = x + [PAD]*(max_len -len(x))
        zs = []
        for z in y:
            z = list(z[:max_string_len])
            zs.append(vocab.token2idx([CLS]+z+[END]) + [vocab.padding_idx]*(max_string_len - len(z)))
        ys.append(zs)
    data = np.transpose(np.array(ys, dtype=np.int64), (1, 0, 2))
    return data


def ArraysToTensor(xs):
    "list of numpy array, each has the same demonsionality"
    x = np.array([ list(x.shape) for x in xs], dtype=np.int64)
    shape = [len(xs)] + list(x.max(axis = 0))
    data = np.zeros(shape, dtype=np.int64)
    for i, x in enumerate(xs):
        slicing_shape = list(x.shape)
        slices = tuple([slice(i, i+1)]+[slice(0, x) for x in slicing_shape])
        data[slices] = x
    return data


def batchify(data, vocabs, unk_rate=0.):
    _tok = ListsToTensor([ [CLS]+x['tok'] for x in data], vocabs['tok'], unk_rate=unk_rate)
    _lem = ListsToTensor([ [CLS]+x['lem'] for x in data], vocabs['lem'], unk_rate=unk_rate)
    _pos = ListsToTensor([ [CLS]+x['pos'] for x in data], vocabs['pos'], unk_rate=unk_rate)
    _ner = ListsToTensor([ [CLS]+x['ner'] for x in data], vocabs['ner'], unk_rate=unk_rate)
    _word_char = ListsofStringToTensor([ [CLS]+x['tok'] for x in data], vocabs['word_char'])

    local_token2idx = [x['token2idx'] for x in data]
    local_idx2token = [x['idx2token'] for x in data]
    _cp_seq = ListsToTensor([ x['cp_seq'] for x in data], vocabs['predictable_concept'], local_token2idx)
    _mp_seq = ListsToTensor([ x['mp_seq'] for x in data], vocabs['predictable_concept'], local_token2idx)

    concept, edge = [], []
    for x in data:
        amr = x['amr']
        concept_i, edge_i, _ = amr.root_centered_sort(vocabs['rel'].priority)
        concept.append(concept_i)
        edge.append(edge_i)

    augmented_concept = [[DUM]+x+[END] for x in concept]

    _concept_char_in = ListsofStringToTensor(augmented_concept, vocabs['concept_char'])[:-1]
    _concept_in      = ListsToTensor(augmented_concept, vocabs['concept'], unk_rate=unk_rate)[:-1]
    _concept_out     = ListsToTensor(augmented_concept, vocabs['predictable_concept'], local_token2idx)[1:]

    out_conc_len, bsz = _concept_out.shape
    _rel = np.full((1+out_conc_len, bsz, out_conc_len), vocabs['rel'].token2idx(PAD), dtype=np.int64)
    # v: [<dummy>, concept_0, ..., concept_l, ..., concept_{n-1}, <end>] u: [<dummy>, concept_0, ..., concept_l, ..., concept_{n-1}]

    for bidx, (x, y) in enumerate(zip(edge, concept)):
        for l, _ in enumerate(y):
            if l > 0:
                # l=1 => pos=l+1=2
                _rel[l+1, bidx, 1:l+1] = vocabs['rel'].token2idx(NIL)
        for v, u, r in x:
            r = vocabs['rel'].token2idx(r)
            _rel[v+1, bidx, u+1] = r

    ret = {'lem':_lem, 'tok':_tok, 'pos':_pos, 'ner':_ner, 'word_char':_word_char, \
           'copy_seq': np.stack([_cp_seq, _mp_seq], -1), \
           'local_token2idx':local_token2idx, 'local_idx2token': local_idx2token, \
           'concept_in':_concept_in, 'concept_char_in':_concept_char_in, \
           'concept_out':_concept_out, 'rel':_rel}

    bert_tokenizer = vocabs.get('bert_tokenizer', None)
    if bert_tokenizer is not None:
        ret['bert_token'] = ArraysToTensor([ x['bert_token'] for x in data])
        ret['token_subword_index'] = ArraysToTensor([ x['token_subword_index'] for x in data])
    return ret


# Note that source can be a filename or a file-type object (ie.. open file or io.StringIO)
# GPU_SIZE = 12000 # okay for 8G memory
class DataLoader(object):
    def __init__(self, vocabs, source, batch_size, for_train, gpu_size=12000):
        self.data = []
        bert_tokenizer = vocabs.get('bert_tokenizer', None)
        for amr, token, lemma, pos, ner in zip(*read_file(source)):
            if for_train:
                _, _, not_ok = amr.root_centered_sort()
                if not_ok or len(token)==0:
                    continue
            cp_seq, mp_seq, token2idx, idx2token = get_concepts(lemma, vocabs['predictable_concept'])
            datum = {'amr':amr, 'tok':token, 'lem':lemma, 'pos':pos, 'ner':ner, \
                     'cp_seq':cp_seq, 'mp_seq':mp_seq,\
                     'token2idx':token2idx, 'idx2token':idx2token}
            if bert_tokenizer is not None:
                bert_token, token_subword_index = bert_tokenizer.tokenize(token)
                datum['bert_token'] = bert_token
                datum['token_subword_index'] = token_subword_index
            self.data.append(datum)
        self.vocabs = vocabs
        self.batch_size = batch_size
        self.train = for_train
        self.unk_rate = 0.
        self.gpu_size = gpu_size

    def set_unk_rate(self, x):
        self.unk_rate = x

    def __iter__(self):
        idx = list(range(len(self.data)))

        if self.train:
            random.shuffle(idx)
            idx.sort(key = lambda x: len(self.data[x]['tok']) + len(self.data[x]['amr']))

        batches = []
        num_tokens, data = 0, []
        for i in idx:
            num_tokens += len(self.data[i]['tok']) + len(self.data[i]['amr'])
            data.append(self.data[i])
            if num_tokens >= self.batch_size:
                sz = len(data)* (2 + max(len(x['tok']) for x in data) + max(len(x['amr']) for x in data))
                if sz > self.gpu_size:
                    # because we only have limited GPU memory
                    batches.append(data[:len(data)//2])
                    data = data[len(data)//2:]
                batches.append(data)
                num_tokens, data = 0, []
        if data:
            sz = len(data)* (2 + max(len(x['tok']) for x in data) + max(len(x['amr']) for x in data))
            if sz > self.gpu_size:
                # because we only have limited GPU memory
                batches.append(data[:len(data)//2])
                data = data[len(data)//2:]
            batches.append(data)

        if self.train:
            random.shuffle(batches)

        for batch in batches:
            yield batchify(batch, self.vocabs, self.unk_rate)
