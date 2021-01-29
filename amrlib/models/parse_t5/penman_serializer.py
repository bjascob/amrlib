import re
import logging
from   enum import Enum
from   collections import Counter
from   tqdm import tqdm
import penman
from   penman.graph import Graph
from   penman.models.noop import NoOpModel
from   ...graph_processing.amr_loading import load_amr_entries

logger = logging.getLogger(__name__)


# Load the penman graph, serialize them and return a dict
def load_and_serialize(fpath, progress=True, max_entries=None):
    entries = load_amr_entries(fpath)[:max_entries]
    serials = {'graphs':[], 'sents':[], 'serials':[]}
    print('Loading and converting', fpath)
    for entry in tqdm(entries, ncols=100, disable=not progress):
        serializer = PenmanSerializer(entry)
        serials['graphs'].append( entry )
        serials['serials'].append(serializer.get_graph_string())
        serials['sents'].append(serializer.get_meta('snt').strip())
    return serials


# Note on parens:
# In AMR format, in the case where the node is only represented by a variable, no parens are used
# to enclose the node.  However, many nodes (all except 2nd use of variable) are something like
# (l0 / league), enclosed in parens
class PenmanSerializer(object):
    INSTANCE = ':instance'
    def __init__(self, gstring):
        self.graph    = penman.decode(gstring, model=NoOpModel())
        # Run the serialization
        self.elements = []              # clear elements list
        self.nodes    = set()           # nodes visited (to prevent recursion)
        self.serialize(self.graph.top)
        self.tokens   = self.elements_to_tokens(self.elements)

    # Return the string where variables are replaced by their values
    def get_graph_string(self):
        return ' '.join(self.tokens)

    # Get the metadata from the graph
    def get_meta(self, key):
        return self.graph.metadata[key]

    # Depth first recurstion of the graph
    # Graph.triples are a list of (source, role, target)
    # note that in the recursive function, node_var is the "target" and could be an attribute/literal value
    def serialize(self, node_var):
        # Apply open paren if this is a variable (not an attrib/literal) and it's the first instance of it
        # If we've seen the variable before, don't insert a new node, just use the reference (ie.. no parens)
        if node_var in self.graph.variables() and node_var not in self.nodes:
            self.elements += ['(', node_var]
        else:
            self.elements += [node_var]
            return      # return if this isn't a variable or if it's the 2nd time we've see the variable
        self.nodes.add(node_var)
        # Loop through all the children of the node and recurse as needed
        children = [t for t in self.graph.triples if t[1] != self.INSTANCE and t[0] == node_var]
        for t in children:
            self.elements.append(t[1])      # add the edge, t[1] is role aka edge
            self.serialize(t[2])            # recurse and add the child (node variable or attrib literal)
        self.elements.append(')')

    # Convert the variables to concepts, but keep the roles and parens the same
    def elements_to_tokens(self, elements):
        # Get the mapping from variables to concepts and then update it (replace)
        # with any enumerated concepts
        var_dict = {t.source:t.target for t in self.graph.instances()}
        var_dict.update(self.get_uid_map())
        tokens   = [var_dict.get(x, x) for x in self.elements]
        return tokens

    # Get a mapping of any non-unique concepts to add a {x} uid to the end
    def get_uid_map(self):
        instances  = self.graph.instances()
        counts     = Counter([t.target for t in instances])
        non_unique = [k for k, c in counts.items() if c > 1]
        uid_map = {}
        for concept in non_unique:
            for i, var in enumerate([t.source for t in instances if t.target == concept]):
                uid_map[var] = '%s_%d' % (concept, i)
        return uid_map


# Error tolerant deserializer
# This deserializers restores the node instances (ie... people, people_01 --> (p / people) and (p2 / people) )
# and attempts to construct a valid AMR graph, ingoring minor errors.
# The output from the Seq-to-Seq model tends to have a lot of small defects that impact the structure.
TType  = Enum('TType', 'paren concept role attrib sep') # Token types
class PenmanDeSerializer(object):
    re_uid = re.compile(r'_\d+$')       # detect _1 as a unique id extension to a concept ie (people_1)
    re_var = re.compile(r'[a-z]\d*')    # use fullmatch to detect variables (single letter 0 or more numbers
    re_ii  = re.compile(r'ii\d*')       # match variables starting with ii
    INSTANCE = ':instance'
    def __init__(self, gstring, gid='x'):
        self.gid           = str(gid)
        self.enumerator    = Counter()
        self.var_dict      = {}         # concepts (with enumeration if needed) mapped to variables
        self.triples       = []
        try:
            self.deserialize(gstring)       # sets self.pgraph and self.gstring
        except:        
            self.gstring = None
            self.pgraph  = None

    def get_pen_graph(self):
        return self.pgraph

    def get_graph_string(self):
        return self.gstring

    # Take a PenamnSerailzer'd graph string (single line, with no metadata) and convert it
    # back to an AMR graph.  Note that at this point, there are no variables in the string.
    # Different instances of concepts are represented by a suffix enumeration, ie.. people_2
    def deserialize(self, gstring):
        node_stack = []     # list of previously instantiated nodes
        node_depth = 0      # number of left parens encountered and not destacked
        triple     = []
        # Tokenize and some logging (the system has a lot of unbalanced parentheses)
        tokens = self.graph_tokenize(gstring)
        # left_parens  = tokens.count('(')
        # right_parens = tokens.count(')')
        # if left_parens != right_parens:
        #     logger.warning('gid=%s has %d left parens and %d right parens' % (self.gid, left_parens, right_parens))
        # Loop through all tokens and parse the string
        for tnum, token in enumerate(tokens):
            #### Big case statement to classify parts of the graph string ####
            ttype = self.token_type(token)
            # Mostly ignored but can be used for error checking
            if token == '(':
                node_depth += 1
            # Find the source for the triple
            elif len(triple) == 0 and ttype == TType.concept:
                # This path should only happen for a new graph. Make a somewhat arbitrary choice to
                # either stop parsing or to clear out the existing triples to prevent disconnected graphs.
                if len(self.triples) > 0:
                    logger.error('gid=%s Initial node constructed when triples not empty.' % (self.gid))
                    if len(self.triples) > len(tokens)/4:    # if > half done (on average ~2 tokens per triple)
                        break
                    else:
                        self.triples = []
                variable, concept, is_new_node = self.get_var_concept(token)
                triple.append(variable)
                if is_new_node:
                    node_stack.append( variable )
                # Some error logging
                if is_new_node and tokens[tnum-1] != '(':
                    logger.warning('gid=%s Missing starting paren for node %s/%s' % (self.gid, variable, concept))
                if not is_new_node and tokens[tnum-1] == '(':
                    logger.warning('gid=%s Start paren present but %s is not a new concept' % (self.gid, concept))
            elif len(triple) == 0 and ttype == TType.role:
                variable = node_stack[-1]
                triple.append(variable)
                triple.append(token)
            # Look for the role (aka edge)
            elif len(triple) == 1 and ttype == TType.role:
                triple.append(token)
            # Look for the target
            elif len(triple) == 2 and ttype == TType.attrib:
                triple.append(token)
            elif len(triple) == 2 and ttype == TType.concept:
                variable, concept, is_new_node = self.get_var_concept(token)
                if is_new_node:
                    node_stack.append( variable )
                # Some error logging
                if is_new_node and tokens[tnum-1] != '(':
                    logger.warning('gid=%s Missing starting paren for node %s/%s' % (self.gid, variable, concept))
                if not is_new_node and tokens[tnum-1] == '(':
                    logger.warning('gid=%s Start paren present but %s is not a new concept' % (self.gid, concept))
                triple.append(variable)
            # De-stack the root nodes based on closing parens, but don't destack past the top var
            # Log an error if we're trying to empty the stack and it's not the very last token
            elif token == ')':
                if len(node_stack) > 1:
                    node_stack.pop()
                    node_depth -= 1
                elif tnum < len(self.triples)-1:
                    logger.warning('gid=%s Trying to destack past top node' % self.gid)
            # Unknown situation (should never get here)
            else:
                logger.warning('gid=%s Unhandled token %s' % (self.gid, token))
            #### Save the triple if complete ####
            if len(triple) == 3:
                self.triples.append( tuple(triple) )
                triple = []
        # Do a little post-processing check on the triples and fix attribs if needed
        # I haven't found instances that requires this but it could be useful
        for i, triple in enumerate(self.triples):
            if triple[1] == self.INSTANCE:
                continue
            target = triple[2]
            # Check if this is a varible
            if self.re_var.fullmatch(target) or self.re_ii.fullmatch(target):
                continue
            # If it's an attrib enforce attribute syntax
            else:
                if (target.startswith('"') and target.endswith('"')) or self.is_num(target)  or \
                   (target in set(['-', '+', 'interrogative', 'imperative', 'expressive'])):
                    continue
                else:
                    new_target = '"' + target.replace('"', '') + '"'
                self.triples[i] = tuple([triple[0], triple[1], new_target])
                logger.warning('gid=%s Replacing attrib %s with %s' % (self.gid, target, new_target))
        # Now convert to a penman graph and then back to a string
        pgraph = Graph(self.triples)
        # Catch malformed graphs, including disconnected ones, incorrectly quoted attibs, etc..
        try:
            self.gstring = penman.encode(pgraph, indent=6, model=NoOpModel())
            self.pgraph  = penman.decode(self.gstring, model=NoOpModel())
        except:
            self.gstring = None
            self.pgraph  = None

    # From the concept return the variable and concept without a uid
    # Note that the first instance from the serializer you get "people" but after that it starts
    # to add uids such as "people_0".  This means not all concept_wuid will actually have an _x
    def get_var_concept(self, concept_wuid):
        if concept_wuid in self.var_dict:
            variable = self.var_dict[concept_wuid]
            is_first = False
        else:
            first = concept_wuid[0].lower() if concept_wuid[0].isalpha() else 'x'
            first = 'ii' if first == 'i' else first     # use ii to avoid issues with "i"
            index = self.enumerator[first]
            self.enumerator[first] += 1
            variable = first if index==0 else '%s%d' % (first, index+1)
            self.var_dict[concept_wuid] = variable
            is_first = True
        concept_only, _ = self.extract_uid(concept_wuid)
        # add an instance definition to triples when the variable is first defined
        if is_first:
            self.triples.append( (variable, self.INSTANCE, concept_only) )
        return variable, concept_only, is_first

    # Extract unique id from concept
    @classmethod
    def extract_uid(cls, concept):
        match = cls.re_uid.search(concept)
        if match is None:
            return concept, None
        stripped = concept[:match.start()]
        uid      = concept[match.start()+1 : match.end()]     # keep uid as a string
        return stripped, uid

    # Check if a string is number (float or int)
    @staticmethod
    def is_num(val):
        try:
            x = float(val)
            return True
        except ValueError:
            return False

    # Helper function token types
    def token_type(self, token):
        if token in set(['(', ')']):
            return TType.paren
        elif token.startswith(':'):
            return TType.role
        elif token in set(['-', '+', 'interrogative', 'imperative', 'expressive']):
            return TType.attrib
        elif token.startswith('"') or token.endswith('"') or token[0].isdigit(): # fault tolerant def
            return TType.attrib
        elif self.is_num(token):
            return TType.attrib
        elif token == '/':
            return TType.sep
        else:
            return TType.concept

    # Tokenize the graph string
    # Quoted literals are the only tricky thing here because in a few cases they can contain other
    # seperator characters like parens, slashes or spaces.
    @staticmethod
    def graph_tokenize(gstring):
        gstring = gstring.strip()
        tokens = []
        sptr = 0
        in_quote  = False
        for ptr, token in enumerate(gstring):
            # Handle quoted literals
            if token == '"':
                if in_quote:    # end quote
                    tokens.append( gstring[sptr:ptr+1] )
                    sptr = ptr + 1
                else:           # begin quote
                    sptr = ptr
                in_quote = not in_quote
            if in_quote:
                continue
            # Break on other seperator characters
            if token == ' ':
                tokens.append( gstring[sptr:ptr] )
                sptr = ptr + 1
            elif token in set(['(', ')', '/']):
                tokens.append( gstring[sptr:ptr] )
                tokens.append( token )
                sptr = ptr + 1
        # Clean-up empty tokens
        tokens = [t.strip() for t in tokens]
        tokens = [t for t in tokens if t]
        return tokens
