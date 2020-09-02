from   nltk.translate.bleu_score import corpus_bleu
from   nltk.tokenize import word_tokenize


# See https://www.nltk.org/api/nltk.translate.html
class BLEUScorer(object):
    def __init__(self):
        pass

    # Take in a tokenized list of refs and hyps and return the bleu score plus the lengths
    def compute_bleu(self, refs, hyps):
        ref_len = self.get_length(refs)
        hyp_len = self.get_length(hyps)
        refs = self.add_ref_dimension(refs) # nltk allows multiple ref per hyp
        bleu = corpus_bleu(refs, hyps)      # even weights=(0.25, 0.25, 0.25, 0.25)
        return bleu, ref_len, hyp_len

    @staticmethod
    def get_length(tokenized_sents):
        length = sum([len(ts) for ts in tokenized_sents])
        return length

    @staticmethod
    def tokenize_strings(strings, space_tokenize=False):
        vals = []
        for string in strings:
            if space_tokenize:
                tokens = [w.lower() for w in string.split()]
            else:
                tokens = [w.lower() for w in word_tokenize(string)]
            vals.append( tokens )
        return vals

    @staticmethod
    def add_ref_dimension(refs):
        refs = [[r] for r in refs]
        return refs
