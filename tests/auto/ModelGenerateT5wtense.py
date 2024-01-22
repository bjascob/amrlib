#!/usr/bin/python3
import sys
sys.path.insert(0, '../..')    # make '..' first in the lib search path
import os
import logging
import unittest
import spacy        # requried or I get an error. From lazy loading? (spacy used to annotate graphs)
import amrlib
from   amrlib.models.generate_xfm.model_input_helper import ModelInputHelper


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

graph02 = '''
# ::id DF-199-194215-653_0484.1
# ::snt I am 24 and a mother of a 2 and a half year old.
# ::tokens ["I", "am", "24", "and", "a", "mother", "of", "a", "2", "and", "a", "half", "year", "old", "."]
# ::ner_tags ["O", "O", "DATE", "O", "O", "O", "O", "O", "DATE", "DATE", "DATE", "DATE", "DATE", "DATE", "O"]
# ::ner_iob ["O", "O", "B", "O", "O", "O", "O", "O", "B", "I", "I", "I", "I", "I", "O"]
# ::pos_tags ["PRP", "VBP", "CD", "CC", "DT", "NN", "IN", "DT", "CD", "CC", "DT", "JJ", "NN", "JJ", "."]
# ::lemmas ["i", "be", "24", "and", "a", "mother", "of", "a", "2", "and", "a", "half", "year", "old", "."]
(a / and
      :op1 (a2 / age-01
            :ARG1 (i / i)
            :ARG2 (t / temporal-quantity
                  :quant 24
                  :unit (y2 / year)))
      :op2 (h / have-rel-role-91
            :ARG0 i
            :ARG1 (p / person
                  :age (t3 / temporal-quantity
                        :quant 2.5
                        :unit (y / year)))
            :ARG2 (m / mother)))
'''

graph03 = '''
(a / and
      :op1 (a2 / age-01
            :ARG1 (i / i)
            :ARG2 (t / temporal-quantity
                  :quant 24
                  :unit (y2 / year)))
      :op2 (h / have-rel-role-91
            :ARG0 i
            :ARG1 (p / person
                  :age (t3 / temporal-quantity
                        :quant 2.5
                        :unit (y / year)))
            :ARG2 (m / mother)))
'''


class ModelGenerateT5wtense(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def testModelInputHelper(self):
        mih1 = ModelInputHelper(graph01)                # string doesn't have annotations
        self.assertTrue(mih1.annotation_performed)
        mih2 = ModelInputHelper(graph02)                # string has annotations
        self.assertFalse(mih2.annotation_performed)
        mih3 = ModelInputHelper(graph02, reannotate=True)
        self.assertTrue(mih3.annotation_performed)
        self.assertEqual(mih1.get_tagged_oneline(), mih2.get_tagged_oneline())
        self.assertEqual(mih1.get_tagged_oneline(), mih3.get_tagged_oneline())
        # Check for unable to annotate becuase there's no sentece metadata
        with self.assertRaises(KeyError) as context:
            mih4 = ModelInputHelper(graph03)
        self.assertTrue('snt', context.exception)

    def testGtoS(self):
        model_dir = os.path.join(amrlib.defaults.data_dir, 'model_generate_t5wtense')
        gtos = amrlib.load_gtos_model(model_dir=model_dir)
        sents, clips = gtos.generate([graph01], disable_progress=True, use_tense=True)
        self.assertEqual(len(sents), 1)
        self.assertTrue(all([c==0 for c in clips]))
        sents, clips = gtos.generate([graph01], disable_progress=True, use_tense=False)
        self.assertEqual(len(sents), 1)
        self.assertTrue(all([c==0 for c in clips]))

if __name__ == '__main__':
    level  = logging.WARNING
    format = '[%(levelname)s %(filename)s ln=%(lineno)s] %(message)s'
    logging.basicConfig(level=level, format=format)

    # run all methods that start with 'test'
    unittest.main()
