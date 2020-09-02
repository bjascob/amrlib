# encoding=utf8
import re
import logging
import random
import json
from   collections import defaultdict
import penman

logger = logging.getLogger(__name__)


# Read an AMR file
# source – a filename or file-like object to read from
def read_file(source):
    # read preprocessed amr file
    token, lemma, pos, ner, amrs = [], [], [], [], []
    graphs = penman.load(source)
    logger.info('read from %s, %d amrs' % (source, len(graphs)))
    for g in graphs:
        # Load the metadata
        token.append(json.loads(g.metadata['tokens']))
        lemma.append(json.loads(g.metadata['lemmas']))
        pos.append(json.loads(g.metadata['pos_tags']))
        ner.append(json.loads(g.metadata['ner_tags']))
        # Build the AMRGraph from the penman graph
        amr_graph = AMRGraph(g)
        amrs.append(amr_graph)
    return amrs, token, lemma, pos, ner


class AMRGraph(object):
    def __init__(self, g):
        self.root  = g.top
        self.nodes = set()
        self.undirected_edges = defaultdict(list)
        self.name2concept = dict()

        # Get the id field for debug
        gid = g.metadata.get('id', '?')[-8:]    # limit string to last 8 characters

        # Loop through all instance triples (name -> the variable name, ie.. n1)
        for name, _, concept in g.instances():
            if _is_attr_form(concept):
                logger.warn('%s has bad instance concept %s %s ' % (gid, name, concept))
            self.name2concept[name] = concept.lower()
            self.nodes.add(name)

        # Loop through all attribute triples
        for concept, rel, value in g.attributes():
            # remove leading colon from penam triple notation
            rel = rel[1:]
            # For attributes penman double-quotes words but the convention here is to use an underscore
            # attributes that are not quoted (just numbers ?) are left as-is
            if '"' in value:
                value = value.replace('"', '') + '_'
            # discard name attributes who's taget value is another name node (ie.. n, n1, etc..)
            if rel == 'name' and discard_regexp.match(value):
                logger.warn('%s has empty name attrib (%s, %s, %s)' % (gid, concept, rel, value))
                continue
            if not _is_attr_form(value):
                logger.warn('%s has bad attribute (%s, %s, %s)' % (gid, concept, rel, value))
                continue
            # Enumerate the attributes to assure uniqueness
            name = "%s_attr_%d" % (value, len(self.name2concept))
            self.name2concept[name] = value     # Keep casing on attribs
            self._add_edge(rel, concept, name)

        # Loop though all edge triples
        # Note that from penman, relations all have a starting colon (ie.. :op1)
        for head, rel, tail in g.edges():
            # remove leading colon from penam triple notation
            rel = rel[1:]
            self._add_edge(rel, head, tail)

    # Add the an edge to the graph, including start and end nodes
    # edges are reversible but attributes are not.
    def _add_edge(self, rel, src, des):
        self.nodes.add(src)
        self.nodes.add(des)
        self.undirected_edges[src].append( (rel, des) )
        self.undirected_edges[des].append( (rel + '_reverse_', src) )

    def root_centered_sort(self, rel_order=None):
        queue = [self.root]
        visited = set(queue)
        step = 0
        while len(queue) > step:
            src = queue[step]
            step += 1
            if src not in self.undirected_edges:
                continue
            random.shuffle(self.undirected_edges[src])
            if rel_order is not None:
                # Do some random thing here for performance enhancement
                if random.random() < 0.5:
                    self.undirected_edges[src].sort(key=lambda x: -rel_order(x[0]) if \
                        (x[0].startswith('snt') or x[0].startswith('op') ) else -1)
                else:
                    self.undirected_edges[src].sort(key=lambda x: -rel_order(x[0]))
            for rel, des in self.undirected_edges[src]:
                if des in visited:
                    continue
                else:
                    queue.append(des)
                    visited.add(des)
        not_connected = len(queue) != len(self.nodes)
        assert (not not_connected)
        name2pos = dict(zip(queue, range(len(queue))))
        visited = set()
        edge = []
        for x in queue:
            if x not in self.undirected_edges:
                continue
            for r, y in self.undirected_edges[x]:
                if y in visited:
                    r = r[:-9] if r.endswith('_reverse_') else r + '_reverse_'
                    edge.append((name2pos[x], name2pos[y], r)) # x -> y: r
            visited.add(x)
        return [self.name2concept[x] for x in queue], edge, not_connected

    # Number of unique tokens in the graph - used in DataLoader
    def __len__(self):
        return len(self.name2concept)


##### Functions to look at words and determine if they fit a specific pattern ####
##### This helps weed out non-conformant triples in the graph.                ####

# True if x is one of the specific words, endswith an underscore or is a number
# In this code attributes (which are double-quoted in penman) have an underscore appended
def _is_attr_form(x):
    return (x in attr_value_set or x.endswith('_') or number_regexp.match(x) is not None)
# If the string should have an instance variable (everything but attributes)
def need_an_instance(x):
    return (not _is_attr_form(x))
# Regex / data for above functions
number_regexp  = re.compile(r'^-?(\d)+(\.\d+)?$')
discard_regexp = re.compile(r'^n(\d+)?$')       # match n, n1, n23
attr_value_set = set(['-', '+', 'interrogative', 'imperative', 'expressive'])  # special case attribs
