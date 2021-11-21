import json
from   unidecode import unidecode
import logging
import requests
from   tqdm import tqdm
import penman

logger = logging.getLogger(__name__)


### Server Setup ##################################################################################
# The online server for this has been unreliable but it's easy to setup a version locally
# Download from... https://sourceforge.net/projects/dbpedia-spotlight/files/
# spotlight/dbpedia-spotlight-1.0.0.jar and 2016-10/en/model/en.tar.gz (dated 2018-02-18) 1.9GB
# tar xzf en.tar.gz
# java -jar dbpedia-spotlight-1.0.0.jar path/to/model/folder/en/ http://localhost:2222/rest
# Instructions at https://github.com/dbpedia-spotlight/dbpedia-spotlight/wiki/Run-from-a-JAR
# but the wget downloads didn't work.  Use the sourceforge download location above
# Note that this will not run with java 11, use java 8 instead
###################################################################################################


# Class for adding :wiki tags to a graph.
# If a cache_fn is supplied, the cache will be used first for lookup
# If the cache lookup fails, the url will be used to query the server if supplied.
# Data retrieve from the URL is saved in the cache so calling save_cache() after lookup
# is a good idea so that when re-run, the url doesn't need to be queried
# url online  url="http://api.dbpedia-spotlight.org/en/annotate"
# url local   url="http://localhost:2222/rest/annotate"
class WikiAdder:
    def __init__(self, url=None, cache_fn=None):
        self.confidence = 0.5   # Spotlight query confidence requirement
        self.url        = url
        self.cache      = self.load_cache(cache_fn) if cache_fn is not None else {}
        # For debug stats
        self.cache_init_sz  = len(self.cache)
        self.wiki_lookups   = 0
        self.cache_hits     = 0
        self.server_queries = 0
        self.server_errors  = 0
        self.wiki_found     = 0
        # Add some manual entries to the cache
        self.manually_update_cache()

    # Print some status
    def get_stat_string(self):
        string  = 'Attempted {:,} wiki lookups\n'.format(self.wiki_lookups)
        string += 'Initial cache size is {:,} entries\n'.format(self.cache_init_sz)
        string += 'There were {:,} cache hits and {:,} server queries\n'.format(\
                   self.cache_hits, self.server_queries)
        string += 'For a total of {:,} non-null (ie.. :wiki -) graph updates\n'.format(\
                  self.wiki_found)
        if self.server_errors > 0:
            string += '!! There were {:,} server query errors !!\n'.format(self.server_errors)
        return string[:-1]  # string final line-feed

    # Load a file, add wiki attribs and save it
    def wikify_file(self, infn, outfn):
        new_graphs = []
        for graph in tqdm(penman.load(infn)):
            new_graph = self.wikify_graph(graph)
            new_graphs.append(new_graph)
        penman.dump(new_graphs, outfn, indent=6)

    # Add a wiki attribute to all nodes with a :name edge and node
    def wikify_graph(self, graph):
        gid = graph.metadata.get('id', '')
        # Check for name attributes.  These shouldn't be present but might.
        for name_attrib in [t for t in graph.attributes() if t.role == ':name']:
            logger.warning('%s has :name attrib in graph %s' % (gid, name_attrib))
        # Find all the name edges and loop through them
        name_edges = [t for t in graph.edges() if t.role == ':name']
        for name_edge in name_edges:
            # Get the associated name string and the parent node to add :wiki to
            name_attribs = [t.target for t in graph.attributes() if t.source == name_edge.target]
            name_attribs = [a.replace('"', '') for a in name_attribs]
            name_string  = ' '.join(name_attribs)
            # This typically does not occur (only 1 instance in LDC2015E86),
            # however generated graphs may have more
            if not name_string:
                logger.warning('%s No name assosiated with the edge %s' % (gid, str(name_edge)))
                continue
            # Lookup the phrase in the spotlight data. .
            wiki_val = self.get_spotlight_wiki_data(name_string)
            if wiki_val is not None:
                wiki_val = '"' + wiki_val + '"'     # string attributes are quoted
                self.wiki_found += 1
            else:
                logger.debug('No wiki data for %s' % name_string)
                wiki_val = '-'                      #  Per AMR spec, a dash is used for no reference
            # Find the index of the parent in the graph.triples
            # The index technically doesn't matter but it may impact the print order
            parent_var = name_edge.source
            parent_triples = [t for t in graph.triples if t[1] == ':instance' and t[0] == parent_var]
            if len(parent_triples) != 1:
                logger.error('%s Graph lookup error for %s returned %s' % (gid, parent_var, parent_triples))
                continue
            index = graph.triples.index(parent_triples[0])
            # Now add this to the graph just after the parent and add an empty epidata entry
            triple = (parent_var, ':wiki', wiki_val)
            graph.triples.insert(index, triple)
            graph.epidata[triple] = []
        return graph

    # Get the wikipedia entry.
    # First try the cache and and if that fails, try the URL server lookup if one is speficied.
    # In the case that there is more than 1 entry returned, use the one with the highest score.
    # This happens with longer phrases where multiple words can produce multiple dbpedia entries.
    def get_spotlight_wiki_data(self, phrase):
        self.wiki_lookups += 1
        if phrase in self.cache:
            self.cache_hits += 1
            return self.cache[phrase]
        # Get the data from the URL and in if there are multiple, use the one with the highest score
        if self.url is not None:
            wiki_pages = self.query_spotlight_wiki_server(phrase)
            if not wiki_pages:
                return None
            # Filter out pages where the "surfaceForm" (aka "text") doesn't exactly match
            # Note that experimentally this appears to help the overall score but it also
            # eliminates some good ones
            # "George W. Bush" returns {'wiki': 'George_W._Bush', 'score': 0.9317, 'text': 'Bush'}
            # "Nuclear Nonproliferation Treaty" = returns..
            # {'wiki': 'Treaty_on_the_Non-Proliferation_of_Nuclear_Weapons', 'score': 1.0, 'text':
            #  'Nonproliferation Treaty'}
            wiki_pages = [p for p in wiki_pages if p['text'].lower() == phrase.lower()]
            if not wiki_pages:
                return None
            # Always keep highest scoring page
            wiki_pages = sorted(wiki_pages, key=lambda x:x['score'])
            wiki       = wiki_pages[-1]['wiki']     # take last, sorted low to high
            self.cache[phrase] = wiki
            return wiki
        return None

    # Query the server with a sentence string (or phrase)
    # This returns a dictionary of every entry found with the sentence string (lowered)
    # mapped to the wikipedia entry.  So this can be called with an entire sentence or simply a phrase.
    # Note that the server is case-sensative and proper nouns must be capitalized correctly.
    # For server status see https://status.dbpedia-spotlight.org/#
    def query_spotlight_wiki_server(self, text):
        self.server_queries += 1
        headers = {'accept': 'application/json'}
        data = {'text':text, 'confidence':self.confidence}
        try:
            req = requests.post(self.url, data=data, headers=headers)
            req.raise_for_status()
            ret = req.content.decode('utf-8')
            sdict = json.loads(ret)
        except:
            self.server_errors += 1
            logger.error('Exception quering for: %s' % text)
            return {}
        # Form a list of the returned pages
        wiki_pages = []
        for res in sdict.get('Resources', []):
            wiki  = res['@URI'].split('/')[-1]   # ie.. http://dbpedia.org/resource/Barack_Obama
            wiki  = unidecode(wiki)
            score = float(res['@similarityScore'])
            text  = res['@surfaceForm']
            wiki_pages.append( {'wiki':wiki, 'score':score, 'text':text} )
        return wiki_pages

    # Get all sentences from an AMR file
    @staticmethod
    def get_sents_from_AMR(infn):
        sents = []
        for graph in penman.load(infn):
            sents.append( graph.metadata['snt'] )
        return sents

    # Load the cache data
    @staticmethod
    def load_cache(cache_fn):
        cache = {}
        try:
            with open(cache_fn) as f:
                cache = json.load(f)
            logger.info('%d entries loaded from cache_fn %s' % (len(cache), cache_fn))
        except:
            logger.warning('Cache file %s does not exist.  Empty cache initialized' % cache_fn)
        return cache

    # Save the cache data
    def save_cache(self, cache_fn):
        with open(cache_fn, 'w') as f:
            json.dump(self.cache, f, indent=4)
        return len(self.cache)

    # Manually add some lookups to the cache that aren't in spotlight
    # This is just a placeholder for a more-extensive list if someone wants to put in the effort
    # It's kind-of cheating but abreviations etc.. really should be in the lookup
    def manually_update_cache(self):
        self.cache['U.S.']    = 'United_States'
        self.cache['U.S.A.']  = 'United_States'
        self.cache['America'] = 'United_States'
        self.cache['West']    = 'Western_world'
