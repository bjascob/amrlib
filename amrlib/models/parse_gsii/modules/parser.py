import torch
from   torch import nn
import torch.nn.functional as F
import math
from   .encoder import WordEncoder, ConceptEncoder
from   .decoder import DecodeLayer
from   .transformer import Transformer, SinusoidalPositionalEmbedding, SelfAttentionMask
from   ..data_loader import ListsToTensor, ListsofStringToTensor
from   ..vocabs import DUM, NIL, PAD
from   ..search import Hypothesis, Beam, search_by_batch
from   ..utils import move_to_device


class Parser(nn.Module):
    def __init__(self, vocabs, word_char_dim, word_dim, pos_dim, ner_dim,
                concept_char_dim, concept_dim,
                cnn_filters, char2word_dim, char2concept_dim,
                embed_dim, ff_embed_dim, num_heads, dropout,
                snt_layers, graph_layers, inference_layers, rel_dim, device,
                pretrained_file=None, bert_encoder=None):
        super(Parser, self).__init__()
        self.vocabs = vocabs
        self.word_encoder = WordEncoder(vocabs, word_char_dim, word_dim, pos_dim, ner_dim,
                            embed_dim, cnn_filters, char2word_dim, dropout, pretrained_file)
        self.concept_encoder = ConceptEncoder(vocabs, concept_char_dim, concept_dim, embed_dim,
                                cnn_filters, char2concept_dim, dropout, pretrained_file)
        self.snt_encoder = Transformer(snt_layers, embed_dim, ff_embed_dim, num_heads, dropout)
        self.graph_encoder = Transformer(graph_layers, embed_dim, ff_embed_dim, num_heads, dropout,
                                with_external=True, weights_dropout=False)
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim, device=device)
        self.word_embed_layer_norm = nn.LayerNorm(embed_dim)
        self.concept_embed_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn_mask = SelfAttentionMask(device=device)
        self.decoder = DecodeLayer(vocabs, inference_layers, embed_dim, ff_embed_dim, num_heads, concept_dim, rel_dim, dropout)
        self.dropout = dropout
        self.probe_generator = nn.Linear(embed_dim, embed_dim)
        self.device = device
        self.bert_encoder = bert_encoder
        if bert_encoder is not None:
            self.bert_adaptor = nn.Linear(768, embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.probe_generator.weight, std=0.02)
        nn.init.constant_(self.probe_generator.bias, 0.)

    def encode_step(self, tok, lem, pos, ner, word_char):
        word_repr = self.embed_scale * self.word_encoder(word_char, tok, lem, pos, ner) + self.embed_positions(tok)
        word_repr = self.word_embed_layer_norm(word_repr)
        word_mask = torch.eq(lem, self.vocabs['lem'].padding_idx)

        word_repr = self.snt_encoder(word_repr, self_padding_mask=word_mask)

        probe = torch.tanh(self.probe_generator(word_repr[:1]))
        word_repr = word_repr[1:]
        word_mask = word_mask[1:]
        return word_repr, word_mask, probe

    def encode_step_with_bert(self, tok, lem, pos, ner, word_char, bert_token, token_subword_index):
        bert_embed, _ = self.bert_encoder(bert_token, token_subword_index=token_subword_index)
        word_repr = self.word_encoder(word_char, tok, lem, pos, ner)
        bert_embed = bert_embed.transpose(0, 1)
        word_repr = word_repr + self.bert_adaptor(bert_embed)
        word_repr = self.embed_scale * word_repr + self.embed_positions(tok)
        word_repr = self.word_embed_layer_norm(word_repr)
        word_mask = torch.eq(lem, self.vocabs['lem'].padding_idx)

        word_repr = self.snt_encoder(word_repr, self_padding_mask=word_mask)

        probe = torch.tanh(self.probe_generator(word_repr[:1]))
        word_repr = word_repr[1:]
        word_mask = word_mask[1:]
        return word_repr, word_mask, probe

    def work(self, data, beam_size, max_time_step, min_time_step=1):
        with torch.no_grad():
            if self.bert_encoder is not None:
                word_repr, word_mask, probe = self.encode_step_with_bert(data['tok'], data['lem'],
                            data['pos'], data['ner'], data['word_char'], data['bert_token'], data['token_subword_index'])
            else:
                word_repr, word_mask, probe = self.encode_step(data['tok'], data['lem'], data['pos'], data['ner'], data['word_char'])

            mem_dict = {'snt_state':word_repr,
                        'snt_padding_mask':word_mask,
                        'probe':probe,
                        'local_idx2token':data['local_idx2token'],
                        'copy_seq':data['copy_seq']}
            init_state_dict = {}
            init_hyp = Hypothesis(init_state_dict, [DUM], 0.)
            bsz = word_repr.size(1)
            beams = [ Beam(beam_size, min_time_step, max_time_step, [init_hyp]) for i in range(bsz)]
            search_by_batch(self, beams, mem_dict)
        return beams

    def prepare_incremental_input(self, step_seq):
        conc = ListsToTensor(step_seq, self.vocabs['concept'])
        conc_char = ListsofStringToTensor(step_seq, self.vocabs['concept_char'])
        conc, conc_char = move_to_device(conc, self.device), move_to_device(conc_char, self.device)
        return conc, conc_char

    def decode_step(self, inp, state_dict, mem_dict, offset, topk):
        step_concept, step_concept_char = inp
        word_repr = snt_state = mem_dict['snt_state']
        word_mask = snt_padding_mask = mem_dict['snt_padding_mask']
        probe = mem_dict['probe']
        copy_seq = mem_dict['copy_seq']
        local_vocabs = mem_dict['local_idx2token']
        _, bsz, _ = word_repr.size()

        new_state_dict = {}

        concept_repr = self.embed_scale * self.concept_encoder(step_concept_char, step_concept) + self.embed_positions(step_concept, offset)
        concept_repr = self.concept_embed_layer_norm(concept_repr)
        for idx, layer in enumerate(self.graph_encoder.layers):
            name_i = 'concept_repr_%d'%idx
            if name_i in state_dict:
                prev_concept_repr = state_dict[name_i]
                new_concept_repr = torch.cat([prev_concept_repr, concept_repr], 0)
            else:
                new_concept_repr = concept_repr

            new_state_dict[name_i] = new_concept_repr
            concept_repr, _, _ = layer(concept_repr, kv=new_concept_repr, external_memories=word_repr, external_padding_mask=word_mask)
        name = 'graph_state'
        if name in state_dict:
            prev_graph_state = state_dict[name]
            new_graph_state = torch.cat([prev_graph_state, concept_repr], 0)
        else:
            new_graph_state = concept_repr
        new_state_dict[name] = new_graph_state
        conc_ll, arc_ll, rel_ll = self.decoder(probe, snt_state, new_graph_state, snt_padding_mask, None, None, copy_seq, work=True)
        for i in range(offset):
            name = 'arc_ll%d'%i
            new_state_dict[name] = state_dict[name]
            name = 'rel_ll%d'%i
            new_state_dict[name] = state_dict[name]
        name = 'arc_ll%d'%offset
        new_state_dict[name] = arc_ll
        name = 'rel_ll%d'%offset
        new_state_dict[name] = rel_ll
        pred_arc_prob = torch.exp(arc_ll)
        arc_confidence = torch.log(torch.max(pred_arc_prob, 1-pred_arc_prob))
        arc_confidence[:,:,0] = 0.
        #pred_arc = torch.lt(pred_arc_prob, 0.5)
        #pred_arc[:,:,0] = 1
        #rel_confidence = rel_ll.masked_fill(pred_arc, 0.).sum(-1, keepdim=True)
        LL = conc_ll + arc_confidence.sum(-1, keepdim=True)# + rel_confidence


        def idx2token(idx, local_vocab):
            if idx in local_vocab:
                return local_vocab[idx]
            return self.vocabs['predictable_concept'].idx2token(idx)

        topk_scores, topk_token = torch.topk(LL.squeeze(0), topk, 1) # bsz x k

        results = []
        for s, t, local_vocab in zip(topk_scores.tolist(), topk_token.tolist(), local_vocabs):
            res = []
            for score, token in zip(s, t):
                res.append((idx2token(token, local_vocab), score))
            results.append(res)

        return new_state_dict, results

    def forward(self, data):
        if self.bert_encoder is not None:
            word_repr, word_mask, probe = self.encode_step_with_bert(data['tok'], data['lem'],
                        data['pos'], data['ner'], data['word_char'], data['bert_token'], data['token_subword_index'])
        else:
            word_repr, word_mask, probe = self.encode_step(data['tok'], data['lem'], data['pos'], data['ner'], data['word_char'])
        concept_repr = self.embed_scale * self.concept_encoder(data['concept_char_in'], data['concept_in']) + \
                            self.embed_positions(data['concept_in'])
        concept_repr = self.concept_embed_layer_norm(concept_repr)
        concept_repr = F.dropout(concept_repr, p=self.dropout, training=self.training)
        concept_mask = torch.eq(data['concept_in'], self.vocabs['concept'].padding_idx)
        attn_mask = self.self_attn_mask(data['concept_in'].size(0))
        for idx, layer in enumerate(self.graph_encoder.layers):
            concept_repr, arc_weight, _ = layer(concept_repr,
                                  self_padding_mask=concept_mask, self_attn_mask=attn_mask,
                                  external_memories=word_repr, external_padding_mask=word_mask,
                                  need_weights ='max')

        graph_target_rel = data['rel'][:-1]
        graph_target_arc = torch.ne(graph_target_rel, self.vocabs['rel'].token2idx(NIL)) # 0 or 1
        graph_arc_mask = torch.eq(graph_target_rel, self.vocabs['rel'].token2idx(PAD))
        graph_arc_loss = F.binary_cross_entropy(arc_weight, graph_target_arc.float(), reduction='none')
        graph_arc_loss = graph_arc_loss.masked_fill_(graph_arc_mask, 0.).sum((0, 2))

        probe = probe.expand_as(concept_repr) # tgt_len x bsz x embed_dim
        concept_loss, arc_loss, rel_loss = self.decoder(probe, word_repr, concept_repr, word_mask, concept_mask, attn_mask, \
                    data['copy_seq'], target=data['concept_out'], target_rel=data['rel'][1:])

        concept_tot = concept_mask.size(0) - concept_mask.float().sum(0)
        concept_loss = concept_loss / concept_tot
        arc_loss = arc_loss / concept_tot
        rel_loss = rel_loss / concept_tot
        graph_arc_loss = graph_arc_loss / concept_tot

        return concept_loss.mean(), arc_loss.mean(), rel_loss.mean(), graph_arc_loss.mean()
