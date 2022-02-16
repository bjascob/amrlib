#!/usr/bin/python3
import setup_run_dir    # Set the working directory and python sys.path to 2 levels above
import os
from   amrlib.utils.logging import silence_penman, setup_logging, WARN, ERROR
from   amrlib.evaluate.smatch_enhanced import get_entries, compute_smatch
from   amrlib.models.parse_xfm.inference import Inference
from   amrlib.models.parse_xfm.penman_serializer import load_and_serialize


if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    setup_logging(logfname='logs/test_model_parse_xfm.log', level=WARN)
    silence_penman()
    device     = 'cuda:0'
    corpus_dir = 'amrlib/data/tdata_xfm/'
    ref_in_fn  = 'test.txt.nowiki'     # 1898 amr entries
    model_dir  = 'amrlib/data/model_parse_xfm_bart_large-v0_1_0'
    gold_fpath = os.path.join(model_dir, 'test-gold.txt')
    pred_fpath = os.path.join(model_dir, 'test-pred.txt')
    num_beams   = 4     # use 4 for formal testing (batch_size=16 for 24GB GPU)
    batch_size  = 16
    max_entries = None  # max test data to generate (use None for everything)

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

    # Save the reference and generated graphs, inserting dummy graphs for that are None
    # Originally I was omitting these graphs but that makes it to test after wikification
    # because the graphs will no longer line up with the original file.
    f_ref = open(gold_fpath, 'w')
    f_gen = open(pred_fpath, 'w')
    print('Saving %s and %s' % (gold_fpath, pred_fpath))
    dummies = 0
    for ref_graph, gen_graph in zip(ref_graphs, gen_graphs):
        # If I didn't get a return, form a dummy graph so the file still aligns with the original
        if gen_graph is None:
            dummies += 1
            gen_graph = '# ::snt dummy graph for deserialization failure.\n()'
        f_ref.write(ref_graph + '\n\n')
        f_gen.write(gen_graph + '\n\n')
    f_ref.close()
    f_gen.close()
    print('Out of %d graphs, %d did not deserialize properly.' % (len(ref_graphs), dummies))
    print()

    # Run smatch
    gold_entries = get_entries(gold_fpath)
    test_entries = get_entries(pred_fpath)
    precision, recall, f_score = compute_smatch(test_entries, gold_entries)
    print('SMATCH -> P: %.3f,  R: %.3f,  F: %.3f' % (precision, recall, f_score))
