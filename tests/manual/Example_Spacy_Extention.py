#!/usr/bin/python3
import setup_run_dir    # run script 2 levels up

# basic spacy extension example code
import amrlib
import spacy
amrlib.setup_spacy_extension()
nlp = spacy.load('en_core_web_sm')
doc = nlp('This is a test of the SpaCy extension. The test has multiple sentences.')
graphs = doc._.to_amr()
for graph in graphs:
    print(graph)
