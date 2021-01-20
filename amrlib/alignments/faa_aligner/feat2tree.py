import re
import sys
import logging

logger = logging.getLogger(__name__)


# global variables used in functions
global_input_id = None
global_input_node_id = None

# Defines
INSTANCE = '/' # shorthand for ':instance'
REST = ':rest' # lumping keyword on the lhs


class Feat(object):
    VARIABLE = -1
    def __init__(self, edge, node, id=None):
        self.edge = edge
        self.node = node
        self.id   = id
        self.alignset = set()
        alignsplit = edge.split('~')
        if len(alignsplit) > 1:
            self.alignset |= set(int(i) for i in alignsplit[1].split('e.')[1].split(','))
            self.edge = alignsplit[0]

    def __str__(self):
        return self.edge + '_' + str(self.node)

    def __eq__(self, other):
        return self.node == other.node and self.edge == other.edge

    def __hash__(self):
        return hash(self.node)


class FeatGraph(object):
    '''amr feature structure is edge-labeled graph'''

    #TODO: can a virtual node be a variable node? virtual ndoe?? variable node???
    VARIABLE = -1

    def __init__(self, id, val, type, feats=None, is_virtual=False, pi='', pi_edge=''):
        self.id = id
        self.val = val
        self.type = type
        self.feats = feats
        self.is_virtual = is_virtual
        self.pi = pi
        self.pi_edge = pi_edge
        # yanggao20130912: change alignset to be a set of tuples, for reentrancy case!
        self.alignset = set()
        alignsplit = val.split('~')
        if len(alignsplit) > 1:
            self.alignset |= set( int(i) for i in alignsplit[1].split('e.')[1].split(',') )
            self.val = alignsplit[0]

    # NOTE: yanggao20130814 added alignset printout
    def pp(self, printed_nodes = set([])):
        '''features:
           1) one-line output
           2) corefering nodes are fully specified once and once only
        '''
        if not self.feats or self in printed_nodes:
            if self.alignset:
                return self.val + '~e.' + ','.join(str(i) for i in sorted(self.alignset))
            else:
                return self.val
        s = self.val + ' '
        feats_str_list = []
        printed_nodes = printed_nodes | set([self])
        for ind, feat in enumerate(self.feats):
            node_repr = feat.node.pp(printed_nodes)
            printed_nodes |= set(feat.node.get_nonterm_nodes())
            if feat.alignset:
                feats_str_list.append(feat.edge + '~e.' + ','.join(str(i) for i in sorted(feat.alignset)) + ' ' + node_repr)
            else:
                feats_str_list.append(feat.edge + ' ' + node_repr)
        s += ' '.join(feats_str_list)
        return '(' + s.strip() + ')'

    def get_nonterm_nodes(self):
        ret = []
        if self.feats:
            ret.append(self)
        for f in self.feats:
            ret.extend(f.node.get_nonterm_nodes())
        return ret

    def assign_id(self):
        global global_input_node_id
        if self.id == None:
            self.id = global_input_node_id
            global_input_node_id += 1
            for f in self.feats:
                f.node.assign_id()

    #TODO: create id when doing this corefy, using a dictionary!! similarly for tree, create
    # dictionary, then for virtual nodes just extend it
    def corefy(self, all_nt):
        '''make the coref i under :ARG1 in
           (w / want :ARG0 (i / i) :ARG1 i)
           pointing to the primary subgraph
           i.e., (i / i) under :ARG0

           identify coref node by these criteria:
           1) has no feats
           2) val same as a nonterm node
           3) not under the INSTANCE arg, since we may
              have (i / i) where the second i is not coref
           4) assumes that property value cannot have the
              form of [a-zA-Z]\d*, so not same as nonterm
        '''
        if not self.feats:
            for nt in all_nt:
                if self.val == nt.val:
                    global global_input_id
                    logger.info('amr %d corefy.  referent triple: (%s %s %s)' % \
                        (global_input_id, self.pi.val, self.pi_edge, self.val))
                    # return a non-recursive copy
                    return FeatGraph(None, self.val, None, [])

        for ind, f in enumerate(self.feats):
            # don't mistake the second i in
            # (w / want :ARG0 (i / i) :ARG1 i) as coref
            if f.edge != INSTANCE:
                self.feats[ind].node = f.node.corefy(all_nt)
        return self


# comment starting with '#', or blank line
COMMENT = re.compile(r'\s*#[^\n]*(\n\s*#[^\n]*)*\n\s*|\n')
NODE_BEGIN_AMR = re.compile(r'\s*\(\s*')
NODE_END_AMR = re.compile(r'\s*\)\s*')
# permissive node end checking: (c / cat)) and (c / cat are both allowed, and will be normalized as (c / cat)
# allow anything inside a pair of double quotes
# like in "16:30" in (:time "16:30"), but not allowing space inside ""
# TODO: yanggao20130816 modified to allow alignment like "jon"~e.6
NODE_AMR = re.compile(r'\s*"[^"]*"\s*|\s*[^\s\(\)/:]+\s*')
# :rest with lookahead: qs.(:rest=[/ increase-01] x0 :ARG1 x1) -> S(qnp.x1 qvp.x0)
# TODO: ygao20130130 relax constraint by comment out following, see if it is ok
REL_NORM = r'\s*:[a-zA-Z][^\s\(]*\s+'
REL_INST = r'\s*%s\s*'%INSTANCE
REL_REST = r'\s*%s\s*(=\s*\[\s*%s\s*(\S+)\s*\]\s*)?'%(REST, INSTANCE) # :rest w/ or w/o lookahead
REL_LABEL_AMR = re.compile(r'%s|%s|%s'%(REL_REST, REL_INST, REL_NORM))


def input_amrparse(s, pos = 0, depth = 0):
    '''return a (pos, FeatGraph) pair
    '''
    comment_match = COMMENT.match(s, pos)
    if comment_match:
        pos = comment_match.end()
        comment_symbol = comment_match.group().strip()

    node_match = NODE_AMR.match(s, pos)
    if node_match:
        pos = node_match.end()
        node_symbol = node_match.group().strip()
        node = FeatGraph(None, node_symbol, None, [])
        return pos, node

    nb_match = NODE_BEGIN_AMR.match(s, pos)
    if nb_match:
        pos = nb_match.end()
        pos, node = input_amrparse(s, pos, depth+1)

        if not node:
            return pos, None

        while pos < len(s):
            ne_match = NODE_END_AMR.match(s, pos)
            if ne_match:
                pos = ne_match.end()
                return pos, node

            rel_match = REL_LABEL_AMR.match(s, pos)
            if rel_match:
                pos = rel_match.end()
                rel_symbol = rel_match.group().strip()
                rel = ''.join(rel_symbol.split())
                pos, newnode = input_amrparse(s, pos, depth+1)
                if not newnode:
                    logger.error('error, return pos %d' % pos)
                    return pos, None
                node.feats.append(Feat(rel, newnode))
                newnode.pi, newnode.pi_edge = node, rel
            else:
                print('does not match ne or rel, pos %d' % pos)
                return pos, None
    return pos, None


def get_alignment(amr, tuples):
    for f in amr.feats:
        for tup in tuples:
            amr_tuple = tuple(tup[:-1])
            alignset = set(int(i) for i in tup[-1].rsplit('e.', 1)[1].split(','))
            if len(amr_tuple) == 2 and (amr.val, f.edge) == amr_tuple:
                f.alignset = alignset
            elif len(amr_tuple) == 3 and (amr.val, f.edge, f.node.val) == amr_tuple:
                f.node.alignset = alignset
        get_alignment(f.node, tuples)


def align(amr_str_lines, align_str_lines):
    global global_input_id, global_input_node_id

    global_input_id = -1
    lines_out = []
    for amr_str, align_str in zip(amr_str_lines, align_str_lines):
        amr_str = amr_str.strip().lower()
        align_str = align_str.strip().lower()
        _, amr = input_amrparse(amr_str)
        global_input_id += 1
        amr.corefy(amr.get_nonterm_nodes())
        global_input_node_id = 0
        amr.assign_id()
        tups = list(tuple(i.split('__')) for i in align_str.split() if re.match(r'.+e\.[\d,]+$', i) )
        get_alignment(amr, tups)
        lines_out.append(amr.pp())
    return lines_out
