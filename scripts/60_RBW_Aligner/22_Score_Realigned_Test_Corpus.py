#!/usr/bin/python3
import setup_run_dir    # Set the working directory and python sys.path to 2 levels above
import penman
from   penman.models.noop import NoOpModel
from   amrlib.evaluate.alignment_scorer import AlignmentScorer


if __name__ == '__main__':
    test_fn = 'amrlib/data/alignments/test_realigned.txt'

    # Load the corpuses
    print('Loading corpus data from', test_fn)
    # Use the NoOpModel to preserve graph ordering
    graphs = penman.load(test_fn, model=NoOpModel())
    print()

    # Score against isi automated alignments
    print('Scoring new alignments against isi machine alignments')
    scorer = AlignmentScorer(graphs, graphs, gold_alignment_key='isi_alignments', test_alignment_key='rbw_alignments')
    scores = scorer.score()
    print(scores)
    print()
