import os
from   tqdm import tqdm
from   collections import Counter
from   .amr_graph import read_file


# Exact a count from a list of list
def make_vocab(batch_seq, char_level=False):
    cnt = Counter()
    for seq in batch_seq:
        cnt.update(seq)     # count once for every item
    if not char_level:
        return cnt
    char_cnt = Counter()
    for x, y in cnt.most_common():
        for ch in list(x):
            char_cnt[ch] += y
    return cnt, char_cnt

# Write the vocab file, sorted by count then by name
def write_vocab(vocab, path):
    with open(path, 'w') as fo:
        vocab = sorted(vocab.items())   # Counter() to list of tuples, sorted by name
        vocab = sorted(vocab, key=lambda x:x[1], reverse=True)  # Now sort by count, reversed
        for name, count in vocab:
            fo.write('%s\t%d\n' % (name, count))

def create_vocabs(train_data, vocab_dir):
    print('Loading', train_data)
    # Read the data file and obtain a list of items, 1 for each AMR entry
    amrs, token, lemma, pos, ner = read_file(train_data)
    # collect concepts and relations
    concepts  = []          # nodes
    relations = []          # edges
    predictable_conc = []
    print('Processing')
    for amr, lem in tqdm(zip(amrs, lemma), total=len(amrs)):
        # run 10 times for random sort to get the priorities of different types of edges
        # Using the root_centered_sort() gives a realistic count of the number of times
        # the vocab is used, which is used for priority later.
        for i in range(10):
            concept, edge, not_ok = amr.root_centered_sort()    # edges are randomly shuffled
            relations.append([e[-1] for e in edge])             # edge is (node-a, node-b, relation)
            if i == 0:                                          # concepts are not shuffled
                concepts.append(concept)
                lexical_concepts = set([l for l in lem] + [l + '_' for l in lem])
                predictable_conc.append([c for c in concept if c not in lexical_concepts])

    # make vocabularies
    token_vocab, token_char_vocab = make_vocab(token, char_level=True)
    lemma_vocab, lemma_char_vocab = make_vocab(lemma, char_level=True)
    pos_vocab                     = make_vocab(pos)
    ner_vocab                     = make_vocab(ner)
    conc_vocab, conc_char_vocab   = make_vocab(concepts, char_level=True)
    predictable_conc_vocab        = make_vocab(predictable_conc)
    rel_vocab                     = make_vocab(relations)

    # Print some stats
    num_pc   = sum(len(x) for x in predictable_conc)
    num_conc = sum(len(x) for x in concepts)
    pct      = 100.*num_pc/num_conc
    print('predictable concept coverage: {:,}/{:,} = {:.1f}%'.format(num_pc, num_conc, pct))

    print ('Saving vocabs to ', vocab_dir)
    write_vocab(token_vocab,            os.path.join(vocab_dir, 'tok_vocab'))
    write_vocab(token_char_vocab,       os.path.join(vocab_dir, 'word_char_vocab'))
    write_vocab(lemma_vocab,            os.path.join(vocab_dir, 'lem_vocab'))
    write_vocab(lemma_char_vocab,       os.path.join(vocab_dir, 'lem_char_vocab'))
    write_vocab(pos_vocab,              os.path.join(vocab_dir, 'pos_vocab'))
    write_vocab(ner_vocab,              os.path.join(vocab_dir, 'ner_vocab'))
    write_vocab(conc_vocab,             os.path.join(vocab_dir, 'concept_vocab'))
    write_vocab(conc_char_vocab,        os.path.join(vocab_dir, 'concept_char_vocab'))
    write_vocab(predictable_conc_vocab, os.path.join(vocab_dir, 'predictable_concept_vocab'))
    write_vocab(rel_vocab,              os.path.join(vocab_dir, 'rel_vocab'))
