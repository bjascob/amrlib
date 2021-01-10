# Evaluation Metrics

amrlib provides a number of evaluation metrics for use in testing.  It's not the library's intent to
provide a large suite of test metrics but wrappers and added functionality to several 3rd party libraries have
been included here to facilitate training and test.


For working examples, look in the sub-directories of the
[scripts directory](https://github.com/bjascob/amrlib/tree/master/scripts).


## SMatch
These functions use the existing [smatch](https://github.com/snowblink14/smatch) library at their core
but add multiprocessing to speed up scoring of large corpuses. In addition, code for
[enhanced/detailed](https://github.com/ChunchuanLv/amr-evaluation-tool-enhanced) sub-scoring is included.
The enhanced sub-scoring is intended to allow the developer to see what the parsing algorithm does best
and what needs to be improved.
```
# Smatch              Standard Smatch score (precision, recall, F1)
# Unlabeled           Compute on the predicted graphs after removing all edge labels
# No WSD              Compute while ignoring Propbank senses (e.g., duck-01 vs duck-02)
# Non_sense_frames    F-score on Propbank frame identification without sense (e.g. duck-00)
# Wikification        F-score on the wikification (:wiki roles)
# Named Ent.          F-score on the named entity recognition (:name roles)
# Negations           F-score on the negation detection (:polarity roles)
# IgnoreVars          Computed by replacing variables with their concepts
# Concepts            F-score on the concept identification task
# Frames              F-score on Propbank frame identification without sense (e.g. duck-01)
# Reentrancy          Computed on reentrant edges only
# SRL                 Computed on :ARG-i roles only
```

Example Usage:
```
from amrlib.evaluate.smatch_enhanced import compute_scores
GOLD='amrlib/data/tdata_gsii/test.txt.features'
PRED='amrlib/data/model_parse_gsii/epoch200.pt.test_generated.wiki'
compute_scores(PRED, GOLD)
```

## BLEU
The bleu_scorer uses [NLTK's bleu_score module](https://www.nltk.org/api/nltk.translate.html#module-nltk.translate.bleu_score).
It is a very thin wrapper over the top of that module and is included here largely to provide the
"standard" method for computing these scores, given that the NLTK module has a number of configuration
parameters that can impact results.

Example Usage:
```
from amrlib.evaluate.bleu_scorer import BLEUScorer
bleu_scorer = BLEUScorer()
bleu_score, ref_len, hyp_len = bleu_scorer.compute_bleu(refs, preds)
print('BLEU score: %5.2f' % (bleu_score*100.))
```
Where `refs` and `preds` are list of list of tokens.  ie.. 1st list dimension is for each sentence in the corpus,
and the 2nd dimension is the list of tokens for the sentence.


## Alignment Scoring
The alignment scorer allows you to get the precision, recall and F1 scores for two lists of alignments.

Example:
The gold_alignments and test_alignments are list of lists (or lists of sets) of the same length.
```
scorer = AlignmentScorer(gold_alignments, test_alignments)
scores = scorer.score()
print(scores)
```
For a complete example see [Score_Alignments](https://github.com/bjascob/amrlib/blob/master/scripts/60_RBW_Aligner/12_Score_Alignments.py).
