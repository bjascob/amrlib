#!/usr/bin/python3
import setup_run_dir    # this import tricks script to run from 2 levels up
import warnings
warnings.simplefilter('ignore')
import os
from   amrlib.utils.logging import setup_logging, WARN, ERROR
from   amrlib.graph_processing.amr_loading import load_amr_entries
from   amrlib.models.generate_t5wtense.inference import Inference


# Get the sentence from an AMR graph string
def get_sentence(graph):
    for line in graph.splitlines():
        if line.startswith('# ::snt'):
            return line[len('# :snt')+1:].strip()
    assert False, 'Error, no sentence info in graph string'


if __name__ == '__main__':
    setup_logging(logfname='logs/generate_t5wtense.log', level=ERROR)
    device     = 'cuda:0'
    model_dir  = 'amrlib/data/model_generate_t5wtense/'
    corpus_dir = 'amrlib/data/tdata_generate_t5wtense/'
    test_fn    = 'test.txt.features.nowiki'             # standard AMR graphs
    # Works using GTX TitanX (12GB)
    # greedy (num_beams=1, batch_size=32) run-time =  4min
    #        (num_beams=8,  batch_size=8) run-time = 16min
    #        (num_beams=16, batch_size=4) run-time = 29min
    batch_size = 4
    num_beams  = 16
    use_tense  = True
    rm_clips   = True

    # Create the filenames based on above parameters
    extension  = '.tagged'  if use_tense else '.nowiki'
    extension += '.clipped' if rm_clips  else '.noclip'
    extension += '.beam' + str(num_beams)
    gen_fn     = 'test.txt.generated' + extension
    ref_fn     = 'test.txt.ref_sents' + extension

    fpath = os.path.join(corpus_dir, test_fn)
    print('Loading test data from', fpath)
    graphs = load_amr_entries(fpath)
    sents  = [get_sentence(g) for g in graphs]

    print('Loading model, tokenizer and data')
    inference = Inference(model_dir, batch_size=batch_size, num_beams=num_beams, device=device)

    print('Generating')
    answers, clips = inference.generate(graphs, disable_progress=False, use_tense=use_tense)

    # Filter out any clipped graphs as invalid tests
    # This will raise the BLEU score
    if rm_clips:
        print('%d graphs were clipped during tokenization and will be removed' % sum(clips))
        assert len(answers) == len(sents) == len(clips)
        answers = [a for a, c in zip(answers, clips) if not c]
        sents   = [s for s, c in zip(sents,   clips) if not c]
    else:
        print('The %d clipped graphs will not be removed' % sum(clips))

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
