#!/usr/bin/python3
import setup_run_dir    # Set the working directory and python sys.path to 2 levels above
import penman
from   penman.models.noop import NoOpModel
from   amrlib.evaluate.alignment_scorer import AlignmentScorer, load_gold_alignments


if __name__ == '__main__':
    if 1:   # dev dataset
        gold_alignments_fn = 'amrlib/alignments/isi_hand_alignments/dev-gold.txt'
        test_amr_fn        = 'amrlib/data/alignments/dev-aligned.txt'
    else:   # test dataset
        gold_alignments_fn = 'amrlib/alignments/isi_hand_alignments/test-gold.txt'
        test_amr_fn        = 'amrlib/data/alignments/test-aligned.txt'

    # Print load alignments
    print('Loading alignments from', gold_alignments_fn)
    gold_alignments, gold_ids = load_gold_alignments(gold_alignments_fn)

    # Load the aligned corpus and extract the data
    print('Loading corpus data from', test_amr_fn)
    pgraphs = penman.load(test_amr_fn, model=NoOpModel())
    test_alignments = [g.metadata['rbw_alignments'].strip().split() for g in pgraphs]
    test_alignments = [a for a in test_alignments if a]
    test_ids = [g.metadata['id'] for g in pgraphs]

    # Sanity check that things match up
    assert len(gold_alignments) == len(test_alignments), '%s != %s' % (len(gold_alignments), len(test_alignments))
    assert len(gold_alignments) == 100, len(gold_alignments)
    for gold_id, test_id in zip(gold_ids, test_ids):
        assert gold_id == test_id, '%s != %s' % (gold_id, test_id)
    print('Gold and Test aligment files match')

    # Score against isi automated alignments
    print('Scoring rule based word alignments against isi hand alignments')
    scorer = AlignmentScorer(gold_alignments, test_alignments)
    print(scorer)
    print()
