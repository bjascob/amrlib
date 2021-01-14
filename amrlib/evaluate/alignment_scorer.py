from   types import SimpleNamespace

# Original alignment file had reference ids one line and alignemnts on the next
# plus some spaces, etc..
def load_gold_alignments(fn):
    ids = []
    alignments = []
    with open(fn) as f:
        for line in f:
            if line.startswith('# ::id'):
                line = line[len('# ::id')+1:].strip()
                ids.append( line.strip() )
            elif line.startswith('# ::alignments'):
                line = line[len('# ::alignments')+1:].strip()
                parts = [p.strip() for p in line.split()]
                parts = [p for p in parts if p]
                alignments.append(parts)
    assert len(alignments) == 100, len(alignments)
    return alignments, ids


# Scire the test alignments against the gold alignments
# gold and test alignments are list of lists (or list of sets)
class AlignmentScorer(object):
    def __init__(self, gold_alignments, test_alignments):
        assert len(gold_alignments) == len(test_alignments), '%s != %s' % (len(gold_alignments), len(test_alignments))
        self.precision_scores = []
        self.recall_scores    = []
        for y_true, y_pred in zip(gold_alignments, test_alignments):
            self._add_score(y_true, y_pred)

    def _add_score(self, y_true, y_pred):
        intersection = len(set(y_pred).intersection(set(y_true)))
        if len(y_pred) > 0:
            self.precision_scores.append(intersection/len(y_pred))
        else:
            self.precision_scores.append(0)
        if len(y_true) > 0:
            self.recall_scores.append(intersection/len(y_true))
        else:
            self.recall_scores.append(0)

    def get_precision_recall_f1(self):
        precision = sum(self.precision_scores)/len(self.precision_scores)
        recall    = sum(self.recall_scores)/len(self.recall_scores)
        if precision + recall > 0:
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            f1 = 0
        return precision, recall, f1

    def __str__(self):
        precision, recall, f1 = self.get_precision_recall_f1()
        string  = ''
        string += 'Precision: {:.2f}   Recall: {:.2f}   F1: {:.2f}'.format(100.*precision, 100.*recall, 100.*f1)
        return string
