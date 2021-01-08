import penman
import re
from   collections import Counter, defaultdict
from   types import SimpleNamespace
import numpy as np
from   .amr_graph import _is_attr_form, need_an_instance


# Note that penman triples typically have a colon in front of the relationship but
# it appears to add these automatically when creating the graph.
class GraphBuilder(object):
    def __init__(self, rel_vocab):
        self.enumerator = Counter()
        self.rel_vocab  = rel_vocab
        self.concepts   = []        # List of node names (concets and attributes)
        self.relations  = []        # list of (target_id, source_id, arc_prob, rel_prob:list(vocab))
        self.names      = []        # names for concepts (1:1) ie..  n1, n2, p1, Ohio@attr4@ , ..
        self.arc_thresh = 0.50      # Threshold of arc probability to add an edge  **1
        # **1: Experimentally 0.5 is about optimal, though increasing to 0.9 doesn't decrease the score
        #      and decreasing to 0.1 only drops the smatch score by 0.014 smatch

    # Convert a list of concepts and relations into a penman graph
    # concept is a list of concepts
    # relation is a list of (target_id, source_id, arc_prob, rel_prob:list(vocab))
    def build(self, concepts, relations):
        self.concepts  = concepts
        self.relations = relations
        self.used_arcs  = defaultdict(set) # keep track of edge names aready seen (key is source_id)
        triples  = self.build_instance_triples()       # add self.names
        triples += self.build_edge_attrib_triples()
        graph    = penman.graph.Graph(triples)
        string   = penman.encode(graph, indent=6)
        # Strip the uniqueness post tag (ie.. 2007@attr1@ -> 2007)
        string = re.sub(r'@attr\d+@', '', string)
        return string

    # Create instance triples from a list of concepts (nodes and attributes)
    # This must be the first call because self.names is set here
    def build_instance_triples(self):
        self.enumerator.clear()
        self.names = []     # unique variable or attribute with tag
        triples = []
        # Loop through the concepts
        for i, concept in enumerate(self.concepts):
            # strings patterns match concept forms and thus require and instance variable
            if need_an_instance(concept):
                # The penman library has an issue parsing concepts with parens or tildes
                # These characters shouldn't be present but parsing errors can lead to this.
                # I found 11 instances in 55,635 training data samples
                # The Smatch scoring AMR reader looks to have an additional issue when it sees a quote in a concept
                concept = concept.replace('(', '')
                concept = concept.replace(')', '')
                concept = concept.replace('~', '')
                concept = concept.replace('"', '')
                if concept != self.concepts[i]:
                    self.concepts[i] = concept
                # get the enumerated graph variable and add a triple for it
                name = self.get_enumerated_var(concept)
                triples.append((name, 'instance', concept))
            # Attributes
            else:
                # AMRGraph.py adds an underscore to the end of any string attributes
                # Trade the underscore for quotes so we can put it back in the penman format
                if concept.endswith('_'):
                    # The penman library has an issue parsing an attribute with a quote inside it
                    # and, practially this probably doesn't make sense anyway, so remove it.
                    name = '"' + concept[:-1].replace('"', '') + '"'
                # Numbers or other things which don't get double-quotes applied to them
                else:
                    name = concept
                # Add a temporary tag to attribute names to gaurentee uniqueness. These will be stripped later.
                name = name + '@attr%d@ ' % i
            self.names.append(name)
        return triples

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

    # Create edge and attribute triples from concepts/names and relations
    def build_edge_attrib_triples(self):
        # Put relations n a little more readable format and create a dictionary of them based on target_id
        rel_dict = defaultdict(list)
        for rel in self.relations:
            target_id, source_id, arc_prob, rel_probs = rel
            entry = SimpleNamespace(target_id=target_id, source_id=source_id, arc_prob=arc_prob, rel_probs=rel_probs)
            rel_dict[target_id].append( entry )
        # Loop through an index for every concepts except the first, to find the best relation
        # Note that this is iterating target id backwards, which is not the way the original code was.
        # This produces much better results when combined with enforcing the rule that ARGx can not be
        # repeated for any source node.  When iterating forward, smatch drops ~0.15 points
        triples = []
        for target_id in range(1, len(self.concepts))[::-1]:
            # Look at all relations attached to this target concept.
            # Add triples for any non-attribute relation with a probability greater than 50%.
            # If none are above 50%, add the best one.
            # For attributes, add the best one (attribs only have 1 source connection)
            best = SimpleNamespace(target_id=None, source_id=None, arc_prob=0, rel_probs=[])
            for entry in rel_dict[target_id]:
                assert entry.target_id == target_id
                assert entry.source_id < entry.target_id
                # Attribtes are never sources, they are always terminal nodes.
                if _is_attr_form(self.concepts[entry.source_id]):
                    continue
                # If greater than a 50% arc probability and it's not an attribute, add a triple for it
                if entry.arc_prob >= self.arc_thresh:
                    if not _is_attr_form(self.concepts[entry.target_id]):
                        triples.append( self.form_relation_triple(entry) )
                # Keep track of the max probabilities
                if entry.arc_prob > best.arc_prob:
                    best = entry
            # If the max probability is less than 50% or if the target is an attibute then the code above
            # didn't add any triples so add the best one.
            if best.arc_prob < self.arc_thresh or _is_attr_form(self.concepts[best.target_id]):
                assert best.target_id == target_id
                triples.append( self.form_relation_triple(best) )
        return triples

    # Form a a triple(source, relation, target) for an edge or attribute triple
    def form_relation_triple(self, entry):
        edge_name = self.edge_name_from_rules(entry)
        # If the edge is a reverse type, form the triple backwards to show this
        if edge_name.endswith('_reverse_'):
            return (self.names[entry.target_id], edge_name[:-9], self.names[entry.source_id])
        else:
            return (self.names[entry.source_id], edge_name,      self.names[entry.target_id])

    # Use some common-sense rules to select the most-probable edge name
    # Generally this is the argmax(rel_probs) but there are cases that are illegal (or at least un-heard of) for AMR
    # Since only the edge_names can be changed, loop through the most probably ones until we find one that passes
    def edge_name_from_rules(self, entry):
        target    = self.concepts[entry.target_id]
        is_attrib = _is_attr_form(target)
        # Rules that exactly dictate the edge name
        # Rule: imperative and expressive attributes always have mode as the edge
        if target in ('imperative', 'expressive') and is_attrib:
            return 'mode'   # edge_name
        # Loop until all rules are satisfied
        edge_name_it = EdgeNameIterator(self.rel_vocab, entry.rel_probs)
        edge_name = edge_name_it.get_next()
        while edge_name_it.was_advanced:
            edge_name_it.was_advanced = False
            # Rule: don't repeat ARGx egdes
            if edge_name.startswith('ARG') and edge_name in self.used_arcs[entry.source_id]:               
                edge_name = edge_name_it.get_next()
            # Rule: edges for attributes should not be reversed (X-of type)
            elif edge_name.endswith('_reverse_') and is_attrib:
                edge_name = edge_name_it.get_next()
            # Rule: domain is never an attribute, the target is always a node
            elif edge_name == 'domain' and is_attrib:
                edge_name = edge_name_it.get_next()
            # Rule: polarity attributes are always have '-' for a value
            elif edge_name == 'polarity' and target != '-':
                edge_name = edge_name_it.get_next()
            # Rule: All "name" edges end lead into "name" nodes (but the reverse is not always true)
            elif edge_name == 'name' and target != 'name':
                edge_name = edge_name_it.get_next()
            # Rule: mode is always an attribute
            elif edge_name == 'mode' and not is_attrib:
                edge_name = edge_name_it.get_next()
        # Keep track of used arcs and don't repeat them for the node
        self.used_arcs[entry.source_id].add(edge_name)
        return edge_name


# Helper class to loop through relation probabilities and get the best / next_best edge name
class EdgeNameIterator(object):
    def __init__(self, rel_vocab, rel_probs):
        self.rel_vocab    = rel_vocab
        self.indices      = np.argsort(rel_probs)[::-1]    # index of the probabilities, sorted high to low
        self.ptr          = 0
        self.was_advanced = False
    def get_next(self):
        index              = self.indices[self.ptr]
        self.ptr          += 1       # let this through an exception if we exhaust all available edges
        self.was_advanced  = True
        return self.rel_vocab.idx2token(index)
