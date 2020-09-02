import os


PAD, UNK, DUM, NIL, END, CLS = '<PAD>', '<UNK>', '<DUMMY>', '<NULL>', '<END>', '<CLS>'


# Note: for the function that saves the vocabs, see create_vocabs.py
def get_vocabs(vocab_dir):
    vocabs = dict()
    vocabs['tok']                 = Vocab(os.path.join(vocab_dir, 'tok_vocab'), 5, [CLS])
    vocabs['lem']                 = Vocab(os.path.join(vocab_dir, 'lem_vocab'), 5, [CLS])
    vocabs['pos']                 = Vocab(os.path.join(vocab_dir, 'pos_vocab'), 5, [CLS])
    vocabs['ner']                 = Vocab(os.path.join(vocab_dir, 'ner_vocab'), 5, [CLS])
    vocabs['predictable_concept'] = Vocab(os.path.join(vocab_dir, 'predictable_concept_vocab'), 5, [DUM, END])
    vocabs['concept']             = Vocab(os.path.join(vocab_dir, 'concept_vocab'), 5, [DUM, END])
    vocabs['rel']                 = Vocab(os.path.join(vocab_dir, 'rel_vocab'), 50, [NIL])
    vocabs['word_char']           = Vocab(os.path.join(vocab_dir, 'word_char_vocab'), 100, [CLS, END])
    vocabs['concept_char']        = Vocab(os.path.join(vocab_dir, 'concept_char_vocab'), 100, [CLS, END])
    return vocabs


class Vocab(object):
    def __init__(self, filename, min_occur_cnt, specials = None):
        idx2token = [PAD, UNK] + (specials if specials is not None else [])
        self._priority = dict()
        num_tot_tokens = 0
        num_vocab_tokens = 0
        with open(filename) as f:
            lines = f.readlines()
        for line in lines:
            try:
                token, cnt = line.rstrip('\n').split('\t')
                cnt = int(cnt)
                num_tot_tokens += cnt
            except:
                print(line)
            if cnt >= min_occur_cnt:
                idx2token.append(token)
                num_vocab_tokens += cnt
            self._priority[token] = int(cnt)
        self.coverage = num_vocab_tokens/num_tot_tokens
        self._token2idx = dict(zip(idx2token, range(len(idx2token))))
        self._idx2token = idx2token
        self._padding_idx = self._token2idx[PAD]
        self._unk_idx = self._token2idx[UNK]

    def priority(self, x):
        return self._priority.get(x, 0)

    @property
    def size(self):
        return len(self._idx2token)

    @property
    def unk_idx(self):
        return self._unk_idx

    @property
    def padding_idx(self):
        return self._padding_idx

    def idx2token(self, x):
        if isinstance(x, list):
            return [self.idx2token(i) for i in x]
        return self._idx2token[x]

    def token2idx(self, x):
        if isinstance(x, list):
            return [self.token2idx(i) for i in x]
        return self._token2idx.get(x, self.unk_idx)
