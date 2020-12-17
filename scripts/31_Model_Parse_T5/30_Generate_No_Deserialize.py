#!/usr/bin/python3
import setup_run_dir    # this import tricks script to run from 2 levels up
import warnings
warnings.simplefilter('ignore')
import os
from   amrlib.utils.logging import silence_penman, setup_logging, WARN, ERROR
from   amrlib.models.parse_t5.inference import Inference
from   amrlib.models.parse_t5.penman_serializer import load_and_serialize


# This code is for debug only
# This will run the model inference (aka generate) but it bypasses the deserialize process so
# these graphs can be saved in raw format.  This makes testing changes to the deserializer
# much easier since generate takes about 40 minutes but deserialization can be done in seconds.
# 
# Note tdata_gsii was created with 30_Model_Parse_GSII/10_Annotate_Corpus.py and 12_RemoveWikiData.py
# This can be changed and the raw LDC data could be used.
if __name__ == '__main__':
    setup_logging(logfname='logs/generate_no_deserialize.log', level=ERROR)
    silence_penman()
    device     = 'cuda:0'
    corpus_dir = 'amrlib/data/tdata_gsii/'
    ref_in_fn  = 'test.txt.features.nowiki'     # 1898 amr entries
    model_dir  = 'amrlib/data/model_parse_t5'
    save_dir   = 'amrlib/data/test_parse_t5'
    gen_out_fn = 'test.txt.generated'
    num_beams   = 4
    batch_size  = 12
    max_entries = None    # max test data to generate (use None for everything)

    # Make the out directory
    os.makedirs(save_dir, exist_ok=True)

    # Load and serialize the reference sentences to parse
    fpath = os.path.join(corpus_dir, ref_in_fn)
    print('Loading test data', fpath)
    entries     = load_and_serialize(fpath)
    ref_sents   = entries['sents'][:max_entries]

    # Load and generate the data
    print('Loading model, tokenizer and data')
    inference = Inference(model_dir, batch_size=batch_size, num_beams=num_beams, device=device, ret_raw_gen=True)
    print('Generating')
    gen_graphs, clips = inference.parse_sents(ref_sents, disable_progress=False)

    # Extract the first (best scoring) graph from the list of beams
    gen_graphs = [gen_graphs[i*num_beams] for i in range(len(ref_sents))]
    assert len(gen_graphs) == len(ref_sents) == len(clips)

   # Save generated file
    fname = os.path.join(save_dir, gen_out_fn)
    print('Saving generated data to', fname)
    with open(fname, 'w') as f:
        for graph, clipped in zip(gen_graphs, clips):
            f.write('%d %s\n' % (int(clipped), graph))
