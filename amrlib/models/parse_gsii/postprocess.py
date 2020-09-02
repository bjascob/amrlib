import penman
import re
from   collections import Counter
import numpy as np
from   .amr_graph import _is_attr_form, need_an_instance


class PostProcessor(object):
    def __init__(self, rel_vocab):
        self.rel_vocab = rel_vocab
        self.enumerator = Counter()

    # Turn concept into a unique variable name (ie.. name -> n, n1, n2,...)
    def get_enumerated_var(self, concept):
        first = concept[0]      # first letter
        if not first.isalpha():
            first = 'x'
        idx = self.enumerator[first]
        self.enumerator[first] += 1
        # de-facto standard is to not append a 0 on the first instance but this
        # seems to cause an issue with my version of the smatch scorer's AMR reader,
        # so for now always append a number since a unique value is all that the spec requires.
        return '%s%d' % (first, idx)

    # res_concept:  list of strings (anything that is a node. ie.. concepts or attributes)
    # res_relation: list of (dep:int, head:int, arc_prob:float, rel_prob:list(vocab))
    # Note that penman triples typically have a colon in front of the relationship but
    # it appears to add these automatically when creating the graph.
    def to_triple(self, res_concept, res_relation):
        self.enumerator.clear()
        triples = []
        names = []
        # Loop through the concepts
        for i, c in enumerate(res_concept):
            # strings patterns match concept forms and thus require and instance variable
            if need_an_instance(c):
                name = self.get_enumerated_var(c)
                triples.append((name, 'instance', c))
            else:
                # attributes. In AMRGraph.py all attributes have an underscore appended
                if c.endswith('_'):
                    name = '"'+c[:-1]+'"'   # these need to be quoted in final graph triples
                # Generally numbers or other things which don't get double-quotes applied to them
                else:
                    name = c
                # Add a tag to attribute names to gaurentee uniqueness. These will be stripped later.
                name = name + '@attr%d@ ' % i
            names.append(name)
        # Loop through the relations (aka roles) and get a list of tuples, keyed by i
        # where each tuple is (head:int, arc_prob:float, relation)
        grouped_relation = dict()
        for i, j, p, r in res_relation:
            r = self.rel_vocab.idx2token(np.argmax(np.array(r)))
            grouped_relation[i] = grouped_relation.get(i, []) + [(j, p, r)]
        # Loop through the concepts again to find the best relation
        for i, c in enumerate(res_concept):
            if i == 0:
                continue
            max_p, max_j, max_r = 0., 0, None
            # Look at all relations attached to this concept.  Add triples for any relation with
            # a probability greater than 50%.  If none are above 50%, just add the best one.
            for j, p, r in grouped_relation[i]:
                assert j < i
                if _is_attr_form(res_concept[j]):
                    continue
                # If greater than a 50% probability of an edge add a triple for it
                if p >= 0.5:
                    if not _is_attr_form(res_concept[i]):
                        if r.endswith('_reverse_'):
                            triples.append((names[i], r[:-9], names[j]))
                        else:
                            triples.append((names[j], r, names[i]))
                # Keep track of the max probabilities
                if p > max_p:
                    max_p, max_j, max_r = p, j, r
            # If the max probability is less than 50%, the code above didn't add any edges
            # so add the best one.
            if max_p < 0.5 or _is_attr_form(res_concept[i]):
                if max_r.endswith('_reverse_'):
                    triples.append((names[i], max_r[:-9], names[max_j]))
                else:
                    triples.append((names[max_j], max_r, names[i]))
        return triples

    # concept is a list of concepts
    # relation is a list of tuples = (i, head_id, arc_prob, rel_prob)
    # where i is the index into the concept list (I think)
    def postprocess(self, concept, relation):
        triples = self.to_triple(concept, relation)
        graph   = penman.graph.Graph(triples)
        string  = penman.encode(graph, indent=6)
        # Strip the uniqueness post tag (ie.. 2007@attr1@ -> 2007)
        string = re.sub(r'@attr\d+@', '', string)
        return string
