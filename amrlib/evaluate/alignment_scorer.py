from   types import SimpleNamespace
import penman


# Graphs need to be in penman graph objects
class AlignmentScorer(object):
    def __init__(self, gold_graphs, test_graphs, **kwargs):
        # Sanity check usage
        assert isinstance(gold_graphs, list)
        assert isinstance(gold_graphs[0], penman.graph.Graph)
        assert isinstance(test_graphs, list)
        assert isinstance(test_graphs[0], penman.graph.Graph)
        # Graphs
        self.gold_graphs = gold_graphs
        self.test_graphs = test_graphs
        # Allow for setting parameters
        self.gold_alignment_key  = kwargs.get('gold_alignment_key',  'gold_alignments')
        self.test_alignment_key  = kwargs.get('test_alignment_key',  'alignments')

    # Create a score for the defined graphs
    def score(self):
        scores = AlignmentScores()
        # Create lookup for alignments
        gold_align_dict = self.get_align_dict(self.gold_graphs, self.gold_alignment_key)
        test_align_dict = self.get_align_dict(self.test_graphs, self.test_alignment_key)
        # Get a list of test to gold alignment pairs
        align_pairs_dict = {}
        for id, gold_align in gold_align_dict.items():
            test_align = test_align_dict[id]
            align_pairs_dict[id] = SimpleNamespace(test=test_align, gold=gold_align)
        # Score Precsion, Recall and F1
        # for a text search, on a set of documents
        #   precision is the number of correct results divided by the number of all returned results.
        #   recall is the number of correct results divided by the number of results that should have been returned.
        # precision = tp / (tp + fp)        precision = |pred intersect gold| / |pred|
        # recall    = tp / (tp + fn)        recall =  = |pred intersect gold| / |gold|
        # Note that I can't verify that this is the correct math for computing precision/recall but, it looks
        # like it per the definition.
        for _, pair in align_pairs_dict.items():
            y_true = pair.gold.split()
            y_pred = pair.test.split()
            scores.tp    += len(set(y_pred).intersection(y_true))
            scores.npred += len(y_pred)
            scores.ntrue += len(y_true)
        return scores

    # Create a dictionary of ids to alignments, given a penman graph and metadata key
    def get_align_dict(self, graphs, align_key):
        align_dict = {}
        for graph in graphs:
            id    = graph.metadata['id'].strip()
            align_dict[id] = graph.metadata[align_key].strip()
        return align_dict


# Simple class for returning scores
class AlignmentScores(object):
    def __init__(self):
        self.tp        = 0
        self.npred     = 0
        self.ntrue     = 0

    def get_precision_recall_f1(self):
        precision = self.tp / self.npred
        recall    = self.tp / self.ntrue
        f1        = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1

    def __str__(self):
        precision, recall, f1 = self.get_precision_recall_f1()
        string  = ''
        string += '{:,} matching alignments out of {:,} predicted and {:,} gold\n'.format(self.tp, self.npred, self.ntrue)
        string += 'Precision: {:.1f}   Recall: {:.1f}   F1: {:.1f}'.format(100.*precision, 100.*recall, 100.*f1)
        return string
