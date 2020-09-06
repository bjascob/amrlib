#!/usr/bin/python3
import setup_run_dir    # this import tricks script to run from 2 levels up
import penman
from   penman.models.noop import NoOpModel  # non-deiverting model
import os
from   amrlib.models.generate_t5.inference import Inference
from   amrlib.graph_processing.amr_loading import load_amr_graph_sent


if __name__ == '__main__':
    corpus_dir = 'amrlib/data/LDC2020T02/'

    print('Loading model and tokenizer')
    model_dir   = 'amrlib/data/model_generate_t5'
    device      = 'cuda:0'  # or 'cpu'
    batch_size  = 8
    num_beams   = 8     # 1 ==> greedy
    num_ret_seq = 1     # 1 ==> return best sentence only

    print('Loading test data')
    fpath = os.path.join(corpus_dir, 'test.txt')
    entries = load_amr_graph_sent(fpath)
    graphs  = entries['graph']
    sents   = entries['sent']
    inference = Inference(model_dir, batch_size=batch_size, num_beams=num_beams, device=device,
                          num_ret_seq=num_ret_seq)

    # Try paraphrasing different AMR graphs
    while True:
        # Get Input
        gnum = input('Enter graph number (or q to quit)> ')
        if gnum.lower().startswith('q'):
            exit()
        # Print and generate
        gnum = int(gnum)
        print('Original:')
        print(sents[gnum])
        print()
        # Get the original graph as a penman object and zero-out the metadata
        #pgraph = penman.decode(graphs[gnum], model=NoOpModel())    # disable deinvert on load
        pgraph = penman.decode(graphs[gnum])
        pgraph.metadata = {}
        # Loop through all variables, keeping the original top first
        tops = sorted(pgraph.variables())
        tops.remove( pgraph.top )
        tops.insert(0, pgraph.top)
        new_graphs = [penman.encode(pgraph, top=t) for t in tops]
        # Get the mapping from top variables to the concept for debug
        var2concept = {t.source:t.target for t in pgraph.instances()}
        # Generate
        print('Generated (first is original top variable):')
        gen_sents, _ = inference.generate(new_graphs, disable_progress=True)
        for sent, top in zip(gen_sents, tops):
            print('top: (%s / %s)' % (top, var2concept[top]))
            print('   ', sent)
        print()
        print('-'*40)
        print()
