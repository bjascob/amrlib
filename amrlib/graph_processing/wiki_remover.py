import os
import logging
from   tqdm import tqdm
import penman
from   .amr_loading import load_amr_entries


logger = logging.getLogger(__name__)


# Remove all :wiki entries from a file
def wiki_remove_file(indir, infn, outdir, outfn):
    graphs = []
    inpath = os.path.join(indir, infn)
    entries = load_amr_entries(inpath)
    for entry in tqdm(entries):
        graph = _process_entry(entry)
        graphs.append(graph)
    outpath = os.path.join(outdir, outfn)
    print('Saving file to ', outpath)
    penman.dump(graphs, outpath, indent=6)


# Remove all :wiki entries from an AMR string
def wiki_remove_graph(entry):
    return _process_entry(entry)


# Take in a single AMR string and return a penman graph
def _process_entry(entry):
    pen = penman.decode(entry)
    # Remove :wiki from the graphs since we want to ignore these 
    triples = [t for t in pen.attributes() if t.role == ':wiki']
    for t in triples:
        try:
            pen.triples.remove(t)
            del pen.epidata[t]
        except:
            logger.error('Unable to remove triple: %s' % (t))
    return pen
