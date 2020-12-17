#!/usr/bin/python3
import setup_run_dir    # run script 2 levels up

# Basic parse / generate example code
import amrlib

print('Example parsing')
stog = amrlib.load_stog_model()
graphs = stog.parse_sents(['This is a test of the system.', 'This is a second sentence.'])
for graph in graphs:
    print(graph)
    print()

print('Generation Example - loading model')
gtos = amrlib.load_gtos_model()
sents, _ = gtos.generate(graphs)
for sent in sents:
    print(sent)
