import os
import json
import logging
from   types import SimpleNamespace
from   collections import Counter
import penman
from   unidecode import unidecode
import blink.main_dense as main_dense

logger = logging.getLogger(__name__)


class WikiAdderBlink:
    blink_logger = logging.getLogger('blink')
    def __init__(self, models_path=None, score_thresh=0.0):
        self.score_thresh = score_thresh    # don't use predictions below this score
        self.models       = None
        if models_path is not None:
            print('Loading BLINK models')
            self.load_models(models_path)

    # Load a file, add wiki attribs and save it
    def wikify_file(self, infpath, outfpath):
        print('Loading', infpath)
        pgraphs = penman.load(infpath)
        winfo_list = self.find_wiki_nodes_for_graphs(pgraphs)
        print('Running BLINK to get wiki values')
        winfo_list = self.predict_blink(winfo_list)
        print('Adding and saving graphs to', outfpath)
        pgraphs = self.add_wiki_to_graphs(pgraphs, winfo_list)
        penman.dump(pgraphs, outfpath, indent=6)

    ###############################################################################################
    #### Wiki node finding in graphs
    ###############################################################################################

    # multi-graph version for find_wiki_nodes() below
    # The returned list is not the same length as the input graphs.
    # Use the graph_idx to collate them as needed.
    def find_wiki_nodes_for_graphs(self, graphs):
        winfo_list = []
        for gidx, graph in enumerate(graphs):
            winfo_list += self.find_wiki_nodes(graph, gidx)
        return winfo_list

    # Find the nodes that could have a wiki tag
    def find_wiki_nodes(self, graph, graph_idx):
        winfo_list = []
        sent = graph.metadata['snt']
        gid  = graph.metadata.get('id', '#' + str(graph_idx))
        # Check for name attributes.  These shouldn't be present but might.
        for name_attrib in [t for t in graph.attributes() if t.role == ':name']:
            logger.warning('%s has :name attrib in graph %s' % (gid, name_attrib))
        # Find all the name edges and loop through them
        name_edges = [t for t in graph.edges() if t.role == ':name']
        for name_edge in name_edges:
            # Get all the :opX operators, sort them by X to extract the multi-word name
            # in all of the test data there are not any name_attribs that have roles other than :opX
            name_attribs = [t for t in graph.attributes() if t.source == name_edge.target]
            name_attribs = sorted(name_attribs, key=lambda t:t.role)
            name_list    = [t.target.replace('"', '') for t in name_attribs if t.role.startswith(':op')]
            if not name_list:
                logger.warning('%s has no name assosiated with the edge %s' % (gid, str(name_edge)))
                continue
            # Try to find the mention in the sentence and if so, split the sentence around it
            # The BLINK system uses all lower-case
            mention = ' '.join(name_list).lower()
            sent = sent.strip().lower()
            mention_found = mention in sent
            if mention_found:
                left_idx      = sent.find(mention)
                right_idx     = left_idx + len(mention)
                context_left  = sent[:left_idx]
                context_right = sent[right_idx:]
            # If exact mention is not in the sentence don't bother splitting.
            # Could do some fuzzy logic to find the split but this seems to work fairly well.
            else:
                context_left  = sent
                context_right = sent
            # Create the wiki information dictionary to be use in the model and for further processing
            winfo = {
                # The following are required keys to run the BLINK model
                'label':        'unknown',      # blink's behavior keys off of this
                'label_id':     -1,             # blink's behavior keys off of this
                'context_left':  context_left,
                'mention':       mention,
                'context_right': context_right,
                # The following are for other internal processing or debug
                'graph_id':      gid,
                'graph_idx':     graph_idx,
                'source_var':    name_edge.source,
                'mention_found': mention_found,
                # These will be filled in by self.predict_blink()
                'wiki_val':      '-',
                'score':         -100.0}
            winfo_list.append(winfo)
        # Check that there aren't multiple source nodes.  This doesn't happen in the AMR3 test
        # or dev data but there are two instances in the training data where it does. Predicted
        # graphs may have some of these.
        sv_counter = Counter([winfo['source_var'] for winfo in winfo_list])
        for svar, count in sv_counter.items():
            if count > 1:
                logger.warning('More than one wiki node for var %s in graph %s' % (svar, gid))
        return winfo_list

    ###############################################################################################
    #### Wiki additions to graphs
    ###############################################################################################

    # Multi-graph version of below
    def add_wiki_to_graphs(self, graphs, winfo_list):
        for winfo in winfo_list:
            gidx = winfo['graph_idx']
            graphs[gidx] = self.add_wiki(graphs[gidx], winfo)
        return graphs

    # Add the wiki nodes
    # Note that graph objects are modified in place
    def add_wiki(self, graph, winfo):
        svar     = winfo['source_var']
        wiki_val = winfo['wiki_val']
        if self.score_thresh is not None and winfo['score'] < self.score_thresh:
            wiki_val = '-'
        # Find the index of the parent in the graph.triples
        # The index technically doesn't matter but it may impact the print order
        parent_triples = [t for t in graph.triples if t[1] == ':instance' and t[0] == svar]
        if len(parent_triples) != 1:
            logger.error('%s Graph lookup error for %s returned %s' % (gid, svar, parent_triples))
        index = graph.triples.index(parent_triples[0])
        # Now add this to the graph just after the parent and add an empty epidata entry
        # string attributes are quoted
        if wiki_val != '-':
            if not wiki_val.startswith('"'):
                wiki_val = '"' + wiki_val
            if not wiki_val.endswith('"'):
                wiki_val = wiki_val + '"'
        triple = (svar, ':wiki', wiki_val)
        graph.triples.insert(index, triple)
        graph.epidata[triple] = []
        return graph

    ###############################################################################################
    #### BLINK related methods
    ###############################################################################################

    def load_models(self, models_path):
        args = self.build_blink_config(models_path)
        self.blink_config = args
        self.models = main_dense.load_models(args, logger=self.blink_logger)

    def predict_blink(self, winfo_list):
        assert self.models is not None
        # return (biencoder_accuracy, recall_at, crossencoder_normalized_accuracy, overall_unormalized_accuracy,
        #         len(winfo_list), predictions, scores) all list of length(winfo_list)
        ret = main_dense.run(self.blink_config, self.blink_logger, *self.models, test_data=winfo_list)
        pred_list, score_list = ret[5], ret[6]
        # Load the returned data back into the wiki info dictionaries
        assert len(winfo_list) == len(pred_list) == len(score_list)
        for winfo, preds, scores in zip(winfo_list, pred_list, score_list):
            preds      = [p for p in preds if not p.startswith('List of')]
            prediction = '"%s"' % unidecode(preds[0]) if preds else '-'
            prediction = prediction.replace(' ', '_')
            winfo['wiki_val'] = prediction
            winfo['score']    = float(scores[0])
        return winfo_list

    def build_blink_config(self, models_path, fast=False):
        blink_config = SimpleNamespace()
        blink_config.interactive         = False
        blink_config.test_entities       = None
        blink_config.test_mentions       = None
        blink_config.top_k               = 10
        blink_config.show_url            = False
        blink_config.fast                = False
        blink_config.biencoder_model     = os.path.join(models_path, "biencoder_wiki_large.bin")
        blink_config.biencoder_config    = os.path.join(models_path, "biencoder_wiki_large.json")
        blink_config.entity_catalogue    = os.path.join(models_path, "entity.jsonl")
        blink_config.entity_encoding     = os.path.join(models_path, "all_entities_large.t7")
        blink_config.crossencoder_model  = os.path.join(models_path, "crossencoder_wiki_large.bin")
        blink_config.crossencoder_config = os.path.join(models_path, "crossencoder_wiki_large.json")
        #faiss_index = 'flat' does not improve scores and takes 2X as long to process
        #faiss_index = None ==> use the entity_encoding model (blink_config.index_path is not used)
        #faiss_index = 'flat' or 'hnsw' then requires index_path (replaces entity encoding model)
        blink_config.faiss_index         = None
        #blink_config.index_path          = os.path.join(models_path, "faiss_flat_index.pkl")
        return blink_config
