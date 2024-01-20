#!/usr/bin/env python3
import setup_run_dir    # this import tricks script to run from 2 levels up
import os
from   nltk.tokenize import word_tokenize
from   amrlib.utils.logging import setup_logging, WARN, ERROR
from   amrlib.graph_processing.amr_loading import load_amr_entries
from   amrlib.models.generate_xfm.inference import Inference
from   amrlib.evaluate.bleu_scorer import BLEUScorer


# Get the sentence from an AMR graph string
def get_sentence(graph):
    for line in graph.splitlines():
        if line.startswith('# ::snt'):
            return line[len('# :snt')+1:].strip()
    assert False, 'Error, no sentence info in graph string'


def score_bleu(preds, refs):
    assert len(preds) == len(refs)
    # Lower-case and word_tokenize
    refs  = [word_tokenize(s.strip().lower()) for s in refs]
    preds = [word_tokenize(s.strip().lower()) for s in preds]
    bleu_scorer = BLEUScorer()
    bleu_score, _, _ = bleu_scorer.compute_bleu(refs, preds)
    return bleu_score


if __name__ == '__main__':
    setup_logging(logfname='logs/infer_generate_xfm.log', level=ERROR)
    device     = 'cuda:0'
    model_dir  = 'amrlib/data/model_generate_xfm'
    corpus_dir = 'amrlib/data/tdata_generate_xfm'
    test_fn    = 'test.txt.features.nowiki'
    # For T5-base With a 24GB GPU
    # num_beams=1  batch_size=64
    # num_beams=16 batch_size=8  (for final testing)
    batch_size = 64
    num_beams  = 1
    use_tense  = True
    rm_clips   = True

    # Create the filenames based on above parameters
    ext  = '.tagged'  if use_tense else '.notag'
    ext += '.clipped' if rm_clips  else '.noclip'
    ext += '.beam' + str(num_beams)
    gen_fn = 'test.txt.generated_sents' + ext
    ref_fn = 'test.txt.reference_sents' + ext

    fpath = os.path.join(corpus_dir, test_fn)
    print('Loading test data from', fpath)
    graphs = load_amr_entries(fpath)
    ref_sents  = [get_sentence(g) for g in graphs]
    print(f'Loaded {len(graphs):,} entries')

    print('Loading model, tokenizer and data')
    inference = Inference(model_dir, batch_size=batch_size, num_beams=num_beams, device=device)

    print('Generating')
    gen_sents, clips = inference.generate(graphs, disable_progress=False, use_tense=use_tense)

    # Filter out any clipped graphs as invalid tests
    # This will raise the BLEU score
    if rm_clips:
        print('%d graphs were clipped during tokenization and will be removed' % sum(clips))
        assert len(gen_sents) == len(ref_sents) == len(clips)
        gen_sents = [a for a, c in zip(gen_sents, clips) if not c]
        ref_sents = [s for s, c in zip(ref_sents, clips) if not c]
    else:
        print('The %d clipped graphs will not be removed' % sum(clips))

    # Save reference file
    fname = os.path.join(model_dir, ref_fn)
    print('Saving original data to', fname)
    with open(fname, 'w') as f:
        for sent in ref_sents:
            f.write('%s\n' % sent)

    # Save generated file
    fname = os.path.join(model_dir, gen_fn)
    print('Saving generated data to', fname)
    with open(fname, 'w') as f:
        for sent in gen_sents:
            f.write('%s\n' % sent)

    # Score the sentences
    bleu_score = score_bleu(gen_sents, ref_sents)
    print('BLEU score: %5.2f' % (bleu_score*100.))
