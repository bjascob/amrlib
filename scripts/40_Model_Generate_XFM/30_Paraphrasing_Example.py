#!/usr/bin/python3
import setup_run_dir    # this import tricks script to run from 2 levels up
import os
import penman
import amrlib
from   amrlib.graph_processing.amr_loading import load_amr_graph_sent


# Count the number of incoming edges to a node
def incoming_edge_count(variable, pgraph):
    incomings = [t for t in pgraph.edges() if t.target==variable]
    return len(incomings)


if __name__ == '__main__':
    corpus_dir = 'amrlib/data/LDC2020T02/'

    print('Loading model and tokenizer')
    model_dir   = 'amrlib/data/model_generate_t5wtense'
    device      = 'cuda:0'  # or 'cpu'
    batch_size  = 8
    num_beams   = 8     # 1 ==> greedy
    num_ret_seq = 1     # 1 ==> return best sentence only

    print('Loading test data')
    fpath = os.path.join(corpus_dir, 'test.txt')
    entries = load_amr_graph_sent(fpath)
    graphs  = entries['graph']
    sents   = entries['sent']
    gtos    = amrlib.load_gtos_model(model_dir, batch_size=batch_size, num_beams=num_beams,
                device=device, num_ret_seq=num_ret_seq)

    # Try paraphrasing different AMR graphs
    while True:
        # Get Input
        gnum = input('Enter graph number (or q to quit)> ')
        if gnum.lower().startswith('q'):
            exit()
        # Print and generate
        gnum = int(gnum)
        print('Original:', sents[gnum])
        print()
        # Get the original graph as a penman object and add back in the sentence to
        # metadata (stripped during loading)
        pgraph = penman.decode(graphs[gnum])
        pgraph.metadata['snt'] = sents[gnum]
        # Loop through all variables and select appropriate candidates for the new top variable
        candidate_tops = pgraph.variables()
        candidate_tops.remove( pgraph.top )
        # (optional) Remove nodes with incoming edges - significantly reduces the number of candidates
        candidate_tops = [v for v in candidate_tops if incoming_edge_count(v, pgraph) == 0]
        # Create the list to try, keeping the original top first
        new_tops = [pgraph.top] + candidate_tops
        new_graphs = [penman.encode(pgraph, top=t) for t in new_tops]
        # Get the mapping from top variables to the concept for debug
        var2concept = {t.source:t.target for t in pgraph.instances()}
        # Generate
        print('Generated (first is original top variable):')
        gen_sents, _ = gtos.generate(new_graphs, disable_progress=True)
        for sent, top in zip(gen_sents, new_tops):
            print('top: (%s / %s)' % (top, var2concept[top]))
            print('   ', sent)
        print()
        print('-'*40)
        print()
