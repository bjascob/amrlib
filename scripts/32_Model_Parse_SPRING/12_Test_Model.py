#!/usr/bin/python3
import setup_run_dir    # this import tricks script to run from 2 levels up
import os
import logging
import torch
import penman
from   penman.models.amr import model as amr_model
from   amrlib.utils.logging import silence_penman, setup_logging, WARN, ERROR
from   amrlib.evaluate.smatch_enhanced import get_entries, compute_smatch
from   amrlib.models.parse_spring.inference import Inference
from   amrlib.models.parse_spring.amr_rw import read_raw_amr_data


if __name__ == '__main__':
    logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)   # skip tokenizer warning
    setup_logging(logfname='logs/parse_spring_generate.log', level=ERROR)
    silence_penman()
    device     = torch.device('cuda:0')
    model_dir   = 'amrlib/data/model_parse_spring'
    model_fn    = 'model.pt'
    test_fns    = 'amrlib/data/amr_annotation_3.0/data/amrs/split/test/*.txt'
    gold_path   = os.path.join(model_dir, 'test-gold.txt')
    pred_path   = os.path.join(model_dir, 'test-pred.txt')
    batch_size  = 32    # number of sentences (train uses number of tokens)
    num_beams   = 5     # 5 is used for formal testing
    max_entries = None  # max test data to generate (use None for everything)

    # Setup the inference model
    print('Loading the model from', os.path.join(model_dir, model_fn))
    inference = Inference(model_dir, model_fn, num_beams=num_beams, batch_size=batch_size, device=device)
    tokenizer = inference.tokenizer
    config    = inference.config

    # Load the data
    print('Loading the dataset')
    graphs_gold = read_raw_amr_data(test_fns, use_recategorization=config['use_recategorization'],
                        dereify=config['dereify'], remove_wiki=config['remove_wiki'])
    graphs_gold = graphs_gold[:max_entries]
    sents       = [g.metadata['snt'] for g in graphs_gold]

    # Create the gold test file
    os.makedirs(os.path.dirname(gold_path), exist_ok=True)
    penman.dump(graphs_gold, gold_path, indent=4, model=amr_model)

    # Run the inference
    print('Generating/testing')
    graphs_gen = inference.parse_sents(sents, return_penman=True, disable_progress=False)
    assert len(graphs_gen) == len(graphs_gold)

    # Detect bad graphs
    # In Penman 1.2.0, metadata does not impact penam.Graph.__eq__()
    num_bad = sum(g == Inference.invalid_graph for g in graphs_gen)
    print('Out of %d graphs, %d did not generate properly.' % (len(graphs_gen), num_bad))

    # Save the final graphs
    print('Generated graphs written to', pred_path)
    penman.dump(graphs_gen, pred_path, indent=4, model=amr_model)

    # Run smatch
    gold_entries = get_entries(gold_path)
    test_entries = get_entries(pred_path)
    precision, recall, f_score = compute_smatch(test_entries, gold_entries)
    print('SMATCH -> P: %.3f,  R: %.3f,  F: %.3f' % (precision, recall, f_score))
