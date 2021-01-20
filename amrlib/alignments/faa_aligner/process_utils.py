import re
import sys
import logging
from   collections import defaultdict
from   .feat2tree import input_amrparse, INSTANCE

logger = logging.getLogger(__name__)


#### stem-4-letters.py ####
def stem_4_letters_word(w):
    return w if w.startswith(':') or (w.startswith('++') and w.endswith('++')) else w[:3]
def stem_4_letters_line(line):
    return ' '.join(stem_4_letters_word(w) for w in line.strip().split())
def stem_4_letters_string(string):
    return '\n'.join(stem_4_letters_line(line) for line in string.splitlines()) + '\n'


# #### stem-4-letters.py ####
def filter_eng_by_stopwords(lines, f_stopwords):
    orig_ind_line_list, out_tok_line_list = [], []
    with open(f_stopwords) as f:
        STOP_SET = set(i.strip() for i in f)
    for line in lines:
        orig_ind_list = []
        out_tok_list = []
        for ind, tok in enumerate(line.strip().split()):
            if tok not in STOP_SET:
                orig_ind_list.append(ind)
                out_tok_list.append(tok)
        orig_ind_line_list.append(' '.join(out_tok_list))
        out_tok_line_list.append(' '.join(str(i) for i in orig_ind_list))
    return orig_ind_line_list, out_tok_line_list


#### get_lineartok_with_rel.py ####
def get_lineartok_with_rel(lines, f_stopwords):
    ind = -1
    def getterminal_except_ne(amr, nt_list):
        nonlocal ind    # allow modification of variable in outer function
        if (not amr.feats):
            ind = ind+1
            if (amr.val not in [nt.val for nt in nt_list]):
                return [(amr.pi.val, amr.pi_edge, amr.val)]
        ret = []
        for f in amr.feats:
            ind = ind+1
            if f.edge == INSTANCE:
                # as "company" in the context of ".. / company :name (.."
                if ':name' in [ff.edge for ff in amr.feats]:
                    continue
                if f.node.val == 'name' and f.node.pi.pi_edge == ':name':
                    continue
                if f.node.val in STOP_SET:
                    continue
                ret.append((amr.val, INSTANCE, f.node.val))
            else:
                if f.edge not in STOP_SET:
                    ret.append((amr.val, f.edge))
                ret.extend(getterminal_except_ne(f.node, nt_list))
        return (ret)
    with open(f_stopwords) as f:
        STOP_SET = set(i.strip() for i in f)
    count = 0
    amr_linear_lines, amr_tuple_lines = [], []
    ds_string2 = ''
    for line in lines:
        line = line.strip().lower()
        count += 1
        _, amr = input_amrparse(line)
        t_list = getterminal_except_ne(amr, amr.get_nonterm_nodes())
        amr_linear_lines.append(' '.join(i[-1] for i in t_list))
        amr_tuple_lines.append(' '.join('__'.join(i) for i in t_list))
    return amr_linear_lines, amr_tuple_lines


#### get_id_mapping_uniq.py ####
def get_id_mapping_uniq(lines_1, lines_2):
    common = set()
    for line1, line2 in zip(lines_1, lines_2):
        l1 = line1.strip().split()
        l2 = line2.strip().split()
        common |= set(l1) & set(l2)
    return list(common)


# map-ibmpos-to-origpos_amr-as-f.py
def map_ibmpos_to_origpos_amr_as_f(origpos_lines, real_lines):
    out_lines = []
    for lnum, (origpos_line, real_line) in enumerate(zip(origpos_lines, real_lines)):
        orig_pos_list = origpos_line.strip().split()
        out_link_list = []
        index_errors  = []
        for link in real_line.strip().split():
            ibmpos_f, ibmpos_e = link.split('-')
            try:
                link_line = ibmpos_f + '-' + orig_pos_list[int(ibmpos_e)]
                out_link_list.append(link_line)
            except:
                index_errors.append(int(ibmpos_e))
        out_lines.append(' '.join(out_link_list))
        if index_errors:
            # A single empty line in tmp.eng.tok.origpos.txt will cause a cascade of errors.  ie.. everything after that point
            # is screwed up.  I'm guessing that it causes something to shift in the aliger logic but I'm not sure where.
            # Check for a missing line in tmp.eng.tok.origpos.txt and for original sentences that are only a period or
            # graphs that are (a / amr-empty)
            # Note that the run-time code (FAA_aligner.align_sents) will handle these correctly
            logger.warning('Indexing error on line %d. Check for empty lines in eng_tok_origpos.txt' % lnum)
    return out_lines


# get_aligned_tuple_amr-as-f_add-align.py
def get_aligned_tuple_amr_as_f_add_align(tuple_str_list, align_str_list):
    out_lines = []
    for tuple_str, align_str in zip(tuple_str_list, align_str_list):
        # e->f
        align_dict = defaultdict(set)
        for align in align_str.split():
            f, e = align.split('-')
            align_dict[int(f)].add(int(e))
        out = []
        tuple_list = tuple_str.split()

        for ind, tuple_one in enumerate(tuple_list):
            concept = tuple_one.split('__')[-1]
            if align_dict[ind]:
                out.append( tuple_one+'__'+'e.'+','.join(str(i) for i in align_dict[ind]) )
            elif concept in ['product','company','person', 'thing'] and ind+2 < len(tuple_list) and \
                        tuple_list[ind+1].split('__')[-1] in [':arg0-of', ':arg1-of', ':arg2-of'] and align_dict[ind+2]:
                out.append( tuple_one+'__'+'e.'+','.join(str(i) for i in align_dict[ind+2]) )
            elif ind-1>=0 and tuple_list[ind-1].split('__')[-1] in ['product','company','person', 'thing'] and \
                        concept in [':arg0-of', ':arg1-of', ':arg2-of'] and ind+1<len(tuple_list) and align_dict[ind+1]:
                out.append( tuple_one+'__'+'e.'+','.join(str(i) for i in align_dict[ind+1]) )
            else:
                out.append( tuple_one )
        out_lines.append(' '.join(out))
    return out_lines


# add_word_pos.py
def add_word_pos(lines):
    out_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            out_lines.append('')
        else:
            out_lines.append(' '.join(word + '_' + str(pos) for pos, word in enumerate(line.split())))
    return out_lines


# giza2isi.py
def giza2isi(lines):
    lines_out = []
    for line in lines:
        l = line.strip().split()
        lines_out.append(' '.join( l[ind] + '-' + l[ind+1] for ind in range(0, len(l)-1, 2) ))
    return lines_out

# swap.py
def swap(lines):
    lines_out = []
    for line in lines:
        l = line.strip().split()
        newl = []
        for i in l:
            pos1, pos2 = i.split('-')
            newl.append(pos2 + '-' + pos1)
        lines_out.append(' '.join(newl))
    return lines_out
