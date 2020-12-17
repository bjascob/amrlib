#!/usr/bin/python3
import setup_run_dir    # this import tricks script to run from 2 levels up
import warnings
warnings.simplefilter('ignore')
import os
from   amrlib.models.generate_t5.inference import Inference
from   amrlib.graph_processing.amr_loading import load_amr_graph_sent


if __name__ == '__main__':
    device     = 'cuda:0'
    corpus_dir = 'amrlib/data/LDC2020T02/'
    model_dir  = 'amrlib/data/model_generate_t5'
    gen_fn     = 'test.txt.generated'
    ref_fn     = 'test.txt.ref_sents'
    # Works using GTX TitanX (12GB)
    # greedy (num_beams=1, batch_size=32) run-time =  3min
    #        (num_beams=8,  batch_size=8) run-time = 15min
    #        (num_beams=16, batch_size=4) run-time = 20min
    batch_size = 4
    num_beams  = 16

    print('Loading test data')
    fpath = os.path.join(corpus_dir, 'test.txt')
    entries = load_amr_graph_sent(fpath)
    graphs  = entries['graph']
    sents   = entries['sent']

    print('Loading model, tokenizer and data')
    inference = Inference(model_dir, batch_size=batch_size, num_beams=num_beams, device=device)

    print('Generating')
    answers, clips = inference.generate(graphs, disable_progress=False)
    print('%d graphs were clipped during tokenization' % sum(clips))

    # Filter out any clipped graphs as invalid tests
    # This will raise the BLEU score
    if 1:
        print('Removing clipped entries from test results')
        assert len(answers) == len(sents) == len(clips)
        answers = [a for a, c in zip(answers, clips) if not c]
        sents   = [s for s, c in zip(sents,   clips) if not c]

    # Save reference file
    fname = os.path.join(model_dir, ref_fn)
    print('Saving original data to', fname)
    with open(fname, 'w') as f:
        for sent in sents:
            f.write('%s\n' % sent)

    # Save generated file
    fname = os.path.join(model_dir, gen_fn)
    print('Saving generated data to', fname)
    with open(fname, 'w') as f:
        for sent in answers:
            f.write('%s\n' % sent)
