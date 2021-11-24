#!/usr/bin/python3
import setup_run_dir    # this import tricks script to run from 2 levels up
import os
from   amrlib.utils.logging import silence_penman, setup_logging, INFO, WARN
from   amrlib.graph_processing.wiki_adder import WikiAdder


# For setup see https://amrlib.readthedocs.io/en/latest/wiki/
if __name__ == '__main__':
    silence_penman()
    setup_logging(logfname='./logs/spotlight_wiki_add.log', level=WARN)
    url       = 'http://localhost:2222/rest/annotate'
    data_dir  = 'amrlib/data/model_parse_spring'
    cache_fn  = os.path.join(data_dir, 'spotlight_wiki.json')
    infn      = os.path.join(data_dir, 'test-pred.txt')
    outfn     = infn + '.wiki'

    wiki = WikiAdder(url=url, cache_fn=cache_fn)
    print('Wikifing', infn)
    wiki.wikify_file(infn, outfn)
    print('Data written to', outfn)
    wiki.save_cache(cache_fn)
    print('cache saved to', cache_fn)
    print()
    print( wiki.get_stat_string() )
    print()
