import re
import json
from   copy import deepcopy
import logging
import penman
from   penman.models.noop import NoOpModel
from   ...graph_processing.annotator import annotate_penman
from   ...graph_processing.amr_loading import split_amr_meta
from   ...alignments.rbw_aligner import RBWAligner

logger = logging.getLogger(__name__)


# Notes: the T5 generator trainer uses load_amr_graph_sent to load the training data so this
#   should work but these graphs are not AMR compliant and penman will not load / parse these
#   correctly
class ModelInputHelper(object):
    # Match any non-whitespace character followed by ~e.
    # Capture group(0)=full-match, group(1)=concept  group2=~e.[comma separated numbers]
    # This is not the same as what's in rwb_aligner.py
    align_re = re.compile(r'([^\s]*)(~e\.\d+[,\d+]*)')

    # Constructor
    # Graph can either be an AMR string or a penman.graph.Graph
    def __init__(self, graph, force_annotate=False):
        # Convert or copy the input graph to penman format
        if isinstance(graph, str):
            pgraph = penman.decode(graph, model=NoOpModel())
        elif isinstance(graph, penman.graph.Graph):
            pgraph = deepcopy(pgraph)
        else:
            raise ValueError('Code requires either a string a penman graph')
        # Annotate if needed (aligner/tagging require annotation)
        is_annotated = all([key in pgraph.metadata for key in ('tokens', 'lemmas', 'pos_tags')])
        if not is_annotated or force_annotate:
            sentence = pgraph.metadata['snt']   # Sanity check required tag.  Throws KeyError if missing
            pgraph = annotate_penman(pgraph)
            self.annotation_performed = True    # for unit-testing and debug
        else:
            self.annotation_performed = False
        # Align the graph.  For simplicity, always do this.
        # If there are existing alignments they need to be removed.
        # See https://penman.readthedocs.io/en/latest/api/penman.surface.html
        if penman.surface.alignments(pgraph) or penman.surface.role_alignments(pgraph):
            for key, items in pgraph.epidata.items():
                pgraph.epidata[key] = [x for x in items if not isinstance(x, penman.surface.AlignmentMarker)]
        pgraph = RBWAligner.from_penman_w_json(pgraph).get_penman_graph()
        # get the graph string and pos tags for the tagger
        self.metadata = pgraph.metadata.copy()
        pos_tags = json.loads(self.metadata['pos_tags'])
        pgraph.metadata = {}
        gstring = penman.encode(pgraph, model=NoOpModel(), indent=6)
        # Tag the graph string
        self.gstring_tagged = self.tag(gstring, pos_tags)

    def get_tagged_with_meta(self):
        gstring = ''
        for k, v in self.metadata.items():
            gstring += '# ::%s %s\n' % (k, v)
        gstring += self.gstring_tagged
        return gstring

    def get_tagged_oneline(self):
        gstring = self.gstring_tagged.replace('\n', '')
        gstring = re.sub(' +', ' ', gstring)      # squeeze multiple spaces into a single
        return gstring

    # Take a annotated and aligned AMR graph string, add POS tags and remove any alignment tags
    def tag(self, gstring, pos_tags):
        # Find all matches and come up with a list of replacements
        replacements = []
        for match in self.align_re.finditer(gstring):
            #concept   = match.group(1)
            align_str = match.group(2)
            # the rbw_aligner should not produce comma separated alignments (it's 1:1)
            tnum = int(align_str[3:])
            tag  = pos_tags[tnum]
            repl = '~' + tag
            replacements.append( (align_str, repl) )    # replace ~e.5 with ~NNP
        # Sort replacements by length so ~e.1 doesn't mess-up ~e.10
        replacements = sorted(replacements, key=lambda x:len(x[0]), reverse=True)
        # Make the replacements
        for (orig_str, new_str) in replacements:
            gstring = gstring.replace(orig_str, new_str)
        return gstring

    # Get rid of any Penn tags appended onto the node names with ~
    # Penn tags are caps all letters except 2 that end in $
    @staticmethod
    def untag(gstring):
        gstring = re.sub(r'~[A-Z$]+', '', gstring)
        return gstring

    # Take an AMR graph string (with metadata) and return the graph portion online on a single line
    @staticmethod
    def gstring_to_oneline(gstring):
        meta_lines, graph_lines = split_amr_meta(gstring)
        gstring = ' '.join(graph_lines)
        gstring = re.sub(' +', ' ', gstring)
        return gstring
