#!/usr/bin/python3
import sys
sys.path.insert(0, '../..')    # make '..' first in the lib search path
import logging
import unittest
import spacy
import amrlib


graph01 = '''
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

# UnitTest creates a separate instance of the class for each test in it.
# The init time doesn't seem to get counted towards the total testing time.
# To avoid loading things multiple times, load in globally and reference it in __init__
# as needed.
SPACY_NLP = spacy.load('en_core_web_sm')
class ModelGenericTypes(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        amrlib.setup_spacy_extension()
        self.nlp = SPACY_NLP

    def testStoG(self):
        stog = amrlib.load_stog_model()
        graphs = stog.parse_sents(['This is a test of the system.'])
        self.assertEqual(len(graphs), 1)

    def testGtoS(self):
        gtos = amrlib.load_gtos_model()
        sents, clips = gtos.generate([graph01], disable_progress=True)
        self.assertEqual(len(sents), 1)

    def testSpaCyDoc(self):
        doc = self.nlp('This is a test of the SpaCy extension.  The test has multiple sentence')
        graphs = doc._.to_amr()
        self.assertEqual(len(graphs), 2)

    def testSpaCySpan(self):
        doc = self.nlp('This is a test of the SpaCy extension.  The test has multiple sentence')
        span = list(doc.sents)[0]   # first sentence only
        graphs = span._.to_amr()
        self.assertEqual(len(graphs), 1)


if __name__ == '__main__':
    level  = logging.WARNING
    format = '[%(levelname)s %(filename)s ln=%(lineno)s] %(message)s'
    logging.basicConfig(level=level, format=format)

    # run all methods that start with 'test'
    unittest.main()
