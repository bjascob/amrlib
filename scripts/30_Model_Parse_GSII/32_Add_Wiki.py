#!/usr/bin/python3
import setup_run_dir    # this import tricks script to run from 2 levels up
import os
from   amrlib.utils.logging import silence_penman, setup_logging, INFO, WARN
from   amrlib.graph_processing.wiki_adder_spotlight import WikiAdderSpotlight


if __name__ == '__main__':
    silence_penman()
    setup_logging(logfname='./logs/spotlight_wiki_add.log', level=WARN)
    url      = 'http://localhost:2222/rest/annotate'
    cache_fn = 'amrlib/data/tdata_gsii/spotlight_wiki.json'
    infn     = 'amrlib/data/model_parse_gsii/epoch200.pt.test_generated'
    outfn    = infn + '.wiki'

    wiki = WikiAdderSpotlight(url=url, cache_fn=cache_fn)
    print('Wikifing', infn)
    wiki.wikify_file(infn, outfn)
    print('Data written to', outfn)
    wiki.save_cache(cache_fn)
    print('cache saved to', cache_fn)
    print()
    print( wiki.get_stat_string() )
    print()
