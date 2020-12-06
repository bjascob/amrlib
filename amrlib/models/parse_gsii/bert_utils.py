import torch
from   transformers import BertTokenizer, BertModel
import numpy as np


class BertEncoderTokenizer(BertTokenizer):
    def __init__(self, *args, **kwargs):
        super(BertEncoderTokenizer, self).__init__(*args, **kwargs)

    def tokenize(self, tokens, split=True):
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        if not split:
            split_tokens = [t if t in self.vocab else '[UNK]' for t in tokens]
            gather_indexes = None
        else:
            split_tokens, _gather_indexes = [], []
            for token in tokens:
                indexes = []
                for i, sub_token in enumerate(self.wordpiece_tokenizer.tokenize(token)):
                    indexes.append(len(split_tokens))
                    split_tokens.append(sub_token)
                _gather_indexes.append(indexes)

            # We only want CLS and tokens (exclude SEP)
            _gather_indexes = _gather_indexes[:-1]
            max_index_list_len = max(len(indexes) for indexes in _gather_indexes)
            gather_indexes = np.zeros((len(_gather_indexes), max_index_list_len))
            for i, indexes in enumerate(_gather_indexes):
                for j, index in enumerate(indexes):
                    gather_indexes[i, j] = index

        token_ids = np.array(self.convert_tokens_to_ids(split_tokens))
        return token_ids, gather_indexes

    def _back_to_txt_for_check(self, token_ids):
        for tokens in token_ids:
            print (self.convert_ids_to_tokens(tokens))


class BertEncoder(BertModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

    def forward(self, input_ids, token_subword_index=None):
        """
        :param input_ids: same as it in BertModel
        :param output_all_encoded_layers: same as it in BertModel
        :param token_subword_index: [batch_size, num_tokens, num_subwords]
        :return:
        """
        # encoded_layers: [batch_size, num_subword_pieces, hidden_size]
        token_type_ids = None
        attention_mask = input_ids.ne(0)

        encoded_layers, pooled_output = super(BertEncoder, self).forward(
            input_ids, attention_mask, token_type_ids, return_dict=False)
        if token_subword_index is None:
            return encoded_layers[:, 1:-1], pooled_output
        else:
            return self.average_pooling(encoded_layers, token_subword_index), pooled_output

    def average_pooling(self, encoded_layers, token_subword_index):
        batch_size, num_tokens, num_subwords = token_subword_index.size()
        batch_index = torch.arange(batch_size).view(-1, 1, 1).type_as(token_subword_index)
        token_index = torch.arange(num_tokens).view(1, -1, 1).type_as(token_subword_index)
        _, num_total_subwords, hidden_size = encoded_layers.size()
        expanded_encoded_layers = encoded_layers.unsqueeze(1).expand(
            batch_size, num_tokens, num_total_subwords, hidden_size)
        # [batch_size, num_tokens, num_subwords, hidden_size]
        token_reprs = expanded_encoded_layers[batch_index, token_index, token_subword_index]
        subword_pad_mask = token_subword_index.eq(0).unsqueeze(3).expand(
            batch_size, num_tokens, num_subwords, hidden_size)
        token_reprs.masked_fill_(subword_pad_mask, 0)
        # [batch_size, num_tokens, hidden_size]
        sum_token_reprs = torch.sum(token_reprs, dim=2)
        # [batch_size, num_tokens]
        num_valid_subwords = token_subword_index.ne(0).sum(dim=2)
        pad_mask = num_valid_subwords.eq(0).long()
        # Add ones to arrays where there is no valid subword.
        divisor = (num_valid_subwords + pad_mask).unsqueeze(2).type_as(sum_token_reprs)
        # [batch_size, num_tokens, hidden_size]
        avg_token_reprs = sum_token_reprs / divisor
        return avg_token_reprs

    def max_pooling(self, encoded_layers, token_subword_index):
        batch_size, num_tokens, num_subwords = token_subword_index.size()
        batch_index = torch.arange(batch_size).view(-1, 1, 1).type_as(token_subword_index)
        token_index = torch.arange(num_tokens).view(1, -1, 1).type_as(token_subword_index)
        _, num_total_subwords, hidden_size = encoded_layers.size()
        expanded_encoded_layers = encoded_layers.unsqueeze(1).expand(
            batch_size, num_tokens, num_total_subwords, hidden_size)
        # [batch_size, num_tokens, num_subwords, hidden_size]
        token_reprs = expanded_encoded_layers[batch_index, token_index, token_subword_index]
        subword_pad_mask = token_subword_index.eq(0).unsqueeze(3).expand(
            batch_size, num_tokens, num_subwords, hidden_size)
        token_reprs.masked_fill_(subword_pad_mask, -float('inf'))
        # [batch_size, num_tokens, hidden_size]
        max_token_reprs, _ = torch.max(token_reprs, dim=2)
        # [batch_size, num_tokens]
        num_valid_subwords = token_subword_index.ne(0).sum(dim=2)
        pad_mask = num_valid_subwords.eq(0).unsqueeze(2).expand(
            batch_size, num_tokens, hidden_size)
        max_token_reprs.masked_fill(pad_mask, 0)
        return max_token_reprs
