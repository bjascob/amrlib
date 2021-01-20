#!/usr/bin/python3
import sys
sys.path.insert(0, '../..')    # make '..' first in the lib search path
import os
import logging
import unittest
from   nltk.tokenize import word_tokenize
import amrlib
from   amrlib.alignments.faa_aligner import FAA_Aligner


test_graphs = '''
# ::id DF-199-194215-653_0484.1 ::date 2013-08-30T09:02:09 ::annotator SDL-AMR-09 ::preferred
# ::snt I am 24 and a mother of a 2 and a half year old.
# ::save-date Tue Apr 29, 2014 ::file DF-199-194215-653_0484_1.txt
(a / and
      :op1 (a2 / age-01
            :ARG1 (i / i)
            :ARG2 (t / temporal-quantity :quant 24
                  :unit (y2 / year)))
      :op2 (h / have-rel-role-91
            :ARG0 i
            :ARG1 (p / person
                  :age (t3 / temporal-quantity :quant 2.5
                              :unit (y / year)))
            :ARG2 (m / mother)))

# ::snt .
(a / amr-empty)

# ::snt the
(a / amr-empty)

# ::snt (-:
(e / emoticon :value "(-:")

# ::snt uh
(a / amr-empty)

# ::snt he
(h / he)

# ::id DF-199-194215-653_0484.1 ::date 2013-08-30T09:02:09 ::annotator SDL-AMR-09 ::preferred
# ::snt I am 24 and a mother of a 2 and a half year old.
# ::save-date Tue Apr 29, 2014 ::file DF-199-194215-653_0484_1.txt
(a / and
      :op1 (a2 / age-01
            :ARG1 (i / i)
            :ARG2 (t / temporal-quantity :quant 24
                  :unit (y2 / year)))
      :op2 (h / have-rel-role-91
            :ARG0 i
            :ARG1 (p / person
                  :age (t3 / temporal-quantity :quant 2.5
                              :unit (y / year)))
            :ARG2 (m / mother)))
'''

def get_test_graphs():
    global graphs
    sents, gstrings = [], []
    for gstring in test_graphs.split('\n\n'):
        for line in gstring.splitlines():
            if line.startswith('# ::snt'):
                sent = line[len('# ::snt')+1:]
                sent = ' '.join(word_tokenize(sent))
        sents.append(sent)
        gstrings.append(gstring)
    return sents, gstrings


class AlignerFAA(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def testModelInputHelper(self):
        sents, gstrings = get_test_graphs()
        aligner = FAA_Aligner()
        amr_surface_aligns, alignment_strings = aligner.align_sents(sents, gstrings)
        self.assertEqual(len(amr_surface_aligns), len(alignment_strings))
        self.assertEqual(len(sents), len(alignment_strings))
        self.assertEqual(alignment_strings[0], alignment_strings[6])    # duplicate alignments


if __name__ == '__main__':
    level  = logging.WARNING
    format = '[%(levelname)s %(filename)s ln=%(lineno)s] %(message)s'
    logging.basicConfig(level=level, format=format)

    # run all methods that start with 'test'
    unittest.main()
