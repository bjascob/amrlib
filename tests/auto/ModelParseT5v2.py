#!/usr/bin/python3
import os
import sys
sys.path.insert(0, '../..')    # make '..' first in the lib search path
import logging
import unittest
import spacy
import amrlib
from   amrlib.defaults import data_dir

# Base classes and relative imports are proving to be problematic so for now, simply copy the code.

# UnitTest creates a separate instance of the class for each test in it so __init__ gets called
# a bunch of times. However, they all seem to run in the same process so globals are shared.
# To avoid loading Spacy multiple times cache it in a global variable.
# For the stog_model, amrlib caches this and since there is only one process it will stay in-memory
# across all unit tests (even ones in other files when run with RunAllUnitTests.py) until explicity
# reloaded with amrlib.load_stog_model(model_dir).
# When the spacy extensions are called the they check to see if a global stog_model is not None, and
# only call the loader if it's not already loaded.
T5V2_LOADED = None      # one-shot to assure amrlib.stog_model is reloaded with this specific model
SPACY_NLP   = None
class ModelParseT5v2(unittest.TestCase):
    model_dir = os.path.join(data_dir, 'model_parse_t5-v0_2_0')
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        amrlib.setup_spacy_extension()
        # Load/cache spacy
        global SPACY_NLP
        if SPACY_NLP is None:
            SPACY_NLP = spacy.load('en_core_web_sm')
        self.nlp  = SPACY_NLP
        # Load model in amrlib (amrlib will cache this itself)
        global T5V2_LOADED
        if T5V2_LOADED is None:
            print('Loading', self.model_dir)
            amrlib.load_stog_model(model_dir=self.model_dir)
            T5V2_LOADED = True
        self.stog = amrlib.stog_model

    def testStoG(self):
        graphs = self.stog.parse_sents(['This is a test of the system.'])
        self.assertEqual(len(graphs), 1)
        # Test that "imperative" can be recognized as a node, not just an attribute
        graphs = self.stog.parse_sents(['Making certain distinctions is imperative in looking back on the past'])
        self.assertNotEqual(graphs[0], None)

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
