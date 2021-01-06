import re
import json
import logging
from   types import SimpleNamespace
import penman
from   penman.models.noop import NoOpModel
from   penman.surface import AlignmentMarker
from   .match_candidates import get_match_candidates


logger = logging.getLogger(__name__)


# Rule Base Word Aligner
class RBWAligner(object):
    # find all strings with ~.eX (alignments)
    # Note that this actually a comma separated list  e.x,y,z .. for every word aligned to this node
    # Note this only works when the alignment is anchored to the end of the line
    # Group 1 is the proceeding word, group 2 is comma separated string
    align_re   = re.compile(r'(.*)~e\.(\d+.*)$')
    # find strings ending in -01 and capture what's before it
    concept_re = re.compile(r'(.*)-\d+$')
    def __init__(self, pen_graph, token_list, lemma_list, **kwargs):
        self.graph          = pen_graph
        self.tokens         = token_list
        self.lemmas         = lemma_list
        self.fuzzy_min_ml   = kwargs.get('fuzzy_min_ml', 4)   # minimum number of characters to look at for fuzzy match
        self.align_str_name = kwargs.get('align_str_name', 'alignments')
        self.align_prefix   = 'e.'                            # part of align_re above so not setable for now
        self.align_words()
        self.add_surface_alignments()
        self.add_alignment_string(self.graph, self.align_str_name)

    # build the aligner from a AMR string that uses json encoded tokens and lemmas (as opposed to space tokenized)
    # Use the NoOpModel to prevent decoding errors.  See https://github.com/goodmami/penman/issues/92
    @classmethod
    def from_string_w_json(cls, graph, token_key='tokens', lemma_key='lemmas', **kwargs):
        assert isinstance(graph, str)
        graph  = penman.decode(graph, model=NoOpModel())
        return cls.from_penman_w_json(graph, token_key, lemma_key, **kwargs)

    # Same as above but with the penman graph object instead of a string
    @classmethod
    def from_penman_w_json(cls, graph, token_key='tokens', lemma_key='lemmas', **kwargs):
        assert isinstance(graph, penman.graph.Graph)
        tokens = [w for w in json.loads(graph.metadata[token_key])]
        lemmas = [w for w in json.loads(graph.metadata[lemma_key])]
        return cls(graph, tokens, lemmas, **kwargs)

    # Get the penman graph object
    def get_penman_graph(self):
        return self.graph

    # Get the graph string (and metadata)
    def get_graph_string(self):
        return penman.encode(self.graph, model=NoOpModel(), indent=6)

    # Remove the surface alignments from the penman graph
    def remove_surface_alignments(self):
        for key, values in self.graph.epidata.items():
            self.graph.epidata[key] = [x for x in values if not isinstance(x, AlignmentMarker)]


    ###########################################################################
    #### Rule Base Word Alignments added to Graph as Surface Aligments
    ###########################################################################

    # Align the words in the graph
    def align_words(self):
        # For debug
        logger.debug('Aligning ' + ' '.join(self.tokens))
        # Get a 2D list of candidate words for each token/lemma in the sentence
        self.wlist_candidates = [get_match_candidates(t, l) for t, l in zip(self.tokens, self.lemmas)]
        self.alignments = [None] * len(self.wlist_candidates)   # index in tokens
        # Loop through all triples in the graph and extract the concept
        for t in self.graph.triples:        # (source, role, target)
            # Get the concept for the triple (this is what's matched to the words in the sentence)
            for tinfo in self.get_triple_info(t):
                # Try to do an exact match and if that files, do a fuzzy one.
                found = self.exact_alignment(tinfo)
                if not found:
                    found = self.fuzzy_alignment(tinfo)
                # Print debug info for the triple/concept not aligned to words in the sentence
                if not found:
                    logger.debug('Concept not found: ' + tinfo.concept )
        # Print debug info for sentence words not aligned to concepts/triples
        if logger.getEffectiveLevel() >= logging.DEBUG:
            missing = [t for t, a in zip(self.tokens, self.alignments) if a is not None]
            if missing:
                logger.debug('Words not aligned to triples: ' + str(missing))

    # Get the associated word (concept, or role) from the triple
    # triples are a tuple of (source, role, target)
    # Returns a list of TInfo objects
    def get_triple_info(self, t):
        # fix issue where penman triple can have None in it
        if None in t:
            return []
        # Instance triples are the concept nodes
        if t[1] == penman.graph.CONCEPT_ROLE:       # CONCEPT_ROLE = ':instance'
            if t[2].lower() == 'multi-sentence':    # special case to skip
                return []
            # remove the word sense from the concept string (ie.. strip -01 from establish-01)
            match = self.concept_re.match(t[2])
            if match is not None:
                concept = match[1].lower()  # match returns [fullmatch, g1] where g1 is what proceeds -X
            else:
                concept = t[2].lower()
            return [TInfo(t, concept, 'concept')]
        # Attribute triples are the constant values nodes with an associated role
        elif t[1] != penman.graph.CONCEPT_ROLE and t[2] not in self.graph.variables():
            if t[1] == ':wiki':     # don't match wiki attributes
                return []
            # Get the target value (ie.. the constant value)
            concept = t[2].lower()
            concept = concept.replace('"', '')      # string attributes are quoted, so remove the quotes
            concept_tinfo = TInfo(t, concept, 'concept')
            # Get the edge role
            role = self.extract_edge_role(t[1])
            if role is None:
                return [concept_tinfo]
            else:
                return [concept_tinfo, TInfo(t, role, 'role')]
        # Edges are the relations between nodes
        else:
            role = self.extract_edge_role(t[1])
            if role is None:
                return []
            return [TInfo(t, role, 'role')]

    # Extract edge words for use matching
    @staticmethod
    def extract_edge_role(edge):
        edge = edge.lower()
        # skip some generic / utility edges that shouldn't match words
        if edge[:4] in (':snt', ':arg', ':mod', ':pos', ':li', ':ord', ':polarity', ':wiki') or \
            edge.startswith(':op') or edge.startswith(':arg'):
            return None
        role = edge[1:]             # remove colon from start of edge
        role = role.split('-')[0]   # remove hyphenates
        return role

    # Check for an exact match between words and a list of candidates
    def exact_alignment(self, tinfo):
        found = False
        for i, candidates in enumerate(self.wlist_candidates):
            for word in candidates:
                if word == tinfo.concept and self.alignments[i] == None:
                    if found:
                        logger.debug('Duplicate exact match for concept: ' + tinfo.concept)
                    else:
                        #logger.debug('Exact match found concept: ' + concept + '  word: ' + word)
                        self.alignments[i] = tinfo
                    found = True
        return found

    # Fuzzy match based on the the number of matching characters between a concept and
    # a list of candidate words
    def fuzzy_alignment(self, tinfo):
        found = False
        # Find the best match length
        match_lengths = []
        for i, candidates in enumerate(self.wlist_candidates):
            ml = [self.match_length(w, tinfo.concept) for w in candidates]
            match_lengths.append( max(ml) )
        max_ml = max(match_lengths)
        # Sort out the good candidates
        if max_ml >= self.fuzzy_min_ml:
            for i in range(len(self.wlist_candidates)):
                if match_lengths[i] == max_ml and self.alignments[i] == None:
                    if found:
                        logger.debug('Duplicate fuzzy matches for concept: ' + tinfo.concept)
                    else:
                        #logger.debug('Fuzzy match found concept: ' + concept + '  word: ' + self.wlist_candidates[i][0])
                        self.alignments[i] = tinfo
                    found = True
        return found

    # Get the length of how many letters in the two strings match
    @staticmethod
    def match_length(string1, string2):
        length = 0
        for t1, t2 in zip(string1, string2):
            if t1==t2: length += 1
            else:      break
        return length

    # Apply aligments to graph
    def add_surface_alignments(self):
        for i, tinfo in enumerate(self.alignments):
            if tinfo is None:
                 continue
            elif tinfo.is_concept():
                self.graph.epidata[tinfo.triple].append(penman.surface.Alignment((i,), prefix=self.align_prefix))
            elif tinfo.is_role():
                self.graph.epidata[tinfo.triple].append(penman.surface.RoleAlignment((i,), prefix=self.align_prefix ))
            else:
                raise RuntimeError('Invalid alignment type')


    ###########################################################################
    #### Read Surface Aligments and Create an AMR metadata alignment string
    ###########################################################################

    # Get the alignment string by recursing the graph and finding all the surface alignments
    # This is a classmethod so that it can be called on a graph, with surface alignments that aren't applied here
    @classmethod
    def add_alignment_string(cls, graph, align_key='alignments'):
        results = cls.get_addresses(graph)
        alignments = []
        # Loop through the results
        for result in results:
            match = cls.align_re.match(result.name)
            if match:
                is_role = True if result.type == 'role' else False
                try:
                    tnums = [int(n) for n in match[2].split(',')]    # alignments are a comma separated list
                except:
                    logger.error('Failed to convert to tnums: %s' % result.name)
                    tnums = []
                for tnum in tnums:
                    sort_key  = cls.form_single_alignment(tnum, result.addr, is_role, True)
                    align_str = cls.form_single_alignment(tnum, result.addr, is_role, False)
                    alignments.append( (sort_key, align_str) )
        # Sort by the token number
        alignments    = sorted(alignments, key=lambda x:x[0])
        align_strings = [x[1] for x in alignments]
        align_string  = ' '.join(align_strings)
        graph.metadata[align_key] = align_string

    # Form a single alignment
    @staticmethod
    def form_single_alignment(tnum, address, is_role, for_sorting):
        if for_sorting:
            string = '%03d-%s' % (tnum, address)    # normal sorting puts '10-' before '2-'
        else:
            string = '%d-%s' % (tnum, address)
        if is_role:
            string += '.r'
        return string

    # Convert the graph to a tree structure and Loop through all branches to get the address
    # of the unique nodes with ~e.X attached
    @classmethod
    def get_addresses(cls, graph):
        results = []
        def add_result(addr, name, type):
            results.append( SimpleNamespace(addr=addr, name=name, type=type) )
        tree = penman.configure(graph, model=NoOpModel())
        for path, branch in cls.walk_tree(tree.node, (1,)):
            # Get the node and attribute addresses
            if penman.tree.is_atomic(branch[1]):    # ==> is None or isinstance(x, (str, int, float))
                address = '.'.join(map(str, path))
                concept = branch[1]
                if concept.startswith('"'):     # Attribute
                    add_result(address, concept, 'attrib')
                else:
                    add_result(address, concept, 'node')
            # Get the edge addresses
            if penman.tree.is_atomic(branch[0]) and branch[0] != '/':
                address   = '.'.join(map(str, path))
                edge_name = branch[0]
                add_result(address, edge_name, 'role')
        return results

    # Modified method from original in penman.tree._walk
    @classmethod
    def walk_tree(cls, node, path):
        var, branches = node
        for i, branch in enumerate(branches):
            if i == 0:
                curpath = path
            else:
                curpath = path + (i,)
            yield (curpath, branch)
            _, target = branch
            if not penman.tree.is_atomic(target):
                yield from cls.walk_tree(target, curpath)

# Helper class for information on triples
class TInfo(object):
    def __init__(self, triple, concept, atype):
        assert atype in ('concept', 'role')
        self.triple  = triple   # penman triple tuple (source, role, target)
        self.concept = concept  # concept or role
        self.atype   = atype    # align to concept (aka node) or role (aka edge)
    def is_concept(self):
        return self.atype == 'concept'
    def is_role(self):
        return self.atype == 'role'
    def __str__(self):
        return '%s / %s / %s' % (self.triple, self.concept, self.atype)
