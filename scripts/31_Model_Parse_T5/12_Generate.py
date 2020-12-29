#!/usr/bin/python3
import setup_run_dir    # this import tricks script to run from 2 levels up
import warnings
warnings.simplefilter('ignore')
import os
from   amrlib.utils.logging import silence_penman, setup_logging, WARN, ERROR
from   amrlib.models.parse_t5.inference import Inference
from   amrlib.models.parse_t5.penman_serializer import load_and_serialize


# Note tdata_gsii was created with 30_Model_Parse_GSII/10_Annotate_Corpus.py and 12_RemoveWikiData.py
# This can be changed.  The corpus doesn't need to be annotated (you can skip running 10_x) but
# wikidata should be removed since the model doesn't produce those tags and these graphs will be
# copied as the reference data to be scored in the next step.
if __name__ == '__main__':
    setup_logging(logfname='logs/parse_t5_generate.log', level=ERROR)
    silence_penman()
    device     = 'cuda:0'
    corpus_dir = 'amrlib/data/tdata_gsii/'
    ref_in_fn  = 'test.txt.features.nowiki'     # 1898 amr entries
    model_dir  = 'amrlib/data/model_parse_t5'
    ref_out_fn = 'test.txt.reference'
    gen_out_fn = 'test.txt.generated'
    # Works using GTX TitanX (12GB)
    # Note that the more beams, the better chance of getting a correctly deserialized graph
    # greedy (num_beams=1, batch_size=32) run-time =  12m
    #        (num_beams=4, batch_size=12) run-time =  50m
    #        (num_beams=8,  batch_size=6) run-time = 1h20
    #        (num_beams=16, batch_size=3) run-time = 2h30m
    num_beams   = 4
    batch_size  = 12
    max_entries = None    # max test data to generate (use None for everything)

    fpath = os.path.join(corpus_dir, ref_in_fn)
    print('Loading test data', fpath)
    entries     = load_and_serialize(fpath)
    ref_graphs  = entries['graphs'][:max_entries]
    ref_serials = entries['serials'][:max_entries]
    ref_sents   = entries['sents'][:max_entries]

    print('Loading model, tokenizer and data')
    inference = Inference(model_dir, batch_size=batch_size, num_beams=num_beams, device=device)

    print('Generating')
    gen_graphs = inference.parse_sents(ref_sents, disable_progress=False)
    assert len(gen_graphs) == len(ref_serials)

    # Save the reference and generated graphs, omitting any that are None
    ref_fname = os.path.join(model_dir, ref_out_fn)
    gen_fname = os.path.join(model_dir, gen_out_fn)
    f_ref = open(ref_fname, 'w')
    f_gen = open(gen_fname, 'w')
    print('Saving %s and %s' % (ref_fname, gen_fname))
    skipped = 0
    for ref_graph, gen_graph in zip(ref_graphs, gen_graphs):
        if gen_graph is None:
            skipped += 1
            continue
        f_ref.write(ref_graph + '\n\n')
        f_gen.write(gen_graph + '\n\n')
    f_ref.close()
    f_gen.close()
    print('Out of %d graphs, skipped %d that did not deserialize properly.' % (len(ref_graphs), skipped))
    print()
