#!/usr/bin/python3
import setup_run_dir    # this import tricks script to run from 2 levels up
import os
from   glob import glob
from   amrlib.graph_processing.amr_loading_raw import load_raw_amr


# Extract the setence and one-line graph from a full amr string
def get_graph_sent(amr_strings):
    entries = {'sent':[], 'graph':[]}
    for entry in amr_strings:
        sent     = None
        gstrings = []
        for line in entry.splitlines():
            line = line.strip()
            if line.startswith('# ::snt'):
                sent = line[len('# ::snt'):].strip()
            if not line.startswith('#'):
                gstrings.append( line )
        if sent and gstrings:
            entries['sent'].append(sent)
            entries['graph'].append(' '.join(gstrings))
    return entries

# Write lines of data to a file
def write_lines(dir, fn, lines):
    fpath = os.path.join(dir, fn)
    print('Writing to', fpath)
    with open(fpath, 'w') as f:
        for line in lines:
            f.write(line + '\n')


# Note that the Hand alignments are for the LDC1 concensus files
# See /home/bjascob/DataRepoTemp/AMR-Data/Hand_Alignments_ISI_LDC2014T12/ldc1_gold_alignments_dev.txt  and _text.txt
# There are 100 entries in each test and dev for the LDC1 data and the Hand Alignments
if __name__ == '__main__':
    amr_dir = 'amrlib/data/amr_annotation_1.0/data/split'
    dev_fp  = 'amrlib/data/amr_annotation_1.0/data/split/dev/amr-release-1.0-dev-consensus.txt'
    test_fp = 'amrlib/data/amr_annotation_1.0/data/split/test/amr-release-1.0-test-consensus.txt'
    out_dir = 'amrlib/data/working_faa_aligner'
    max_entries = 200   # 100 entries in each dev and test

    os.makedirs(out_dir, exist_ok=True)

    # Get all the amr files and put dev-consensus.txt on top, followed by test-consensus.txt
    # to make scoring easier
    fpaths = [y for x in os.walk(amr_dir) for y in glob(os.path.join(x[0], '*.txt'))]
    fpaths = sorted([fp for fp in fpaths if fp not in (dev_fp, test_fp)])
    fpaths = [dev_fp, test_fp] + fpaths

    # Load all the entries
    print('Loading data')
    sents, gstrings = [], []
    for fpath in fpaths:
        amr_strings = load_raw_amr(fpath)
        entries = get_graph_sent(amr_strings)
        #entries = load_amr_graph_sent(fpath)
        # Append the data
        # Filter "(a / amr-empty)" in amr-release-1.0-proxy.txt that might be causing issues
        # So long as this is above the dev/test data (ends at index 200) it won't mess-up scoring
        for sent, graph in zip(entries['sent'], entries['graph']):
            if sent == '.':
                print('Removed empty entry at index %d from %s' % (len(sents), fpath))
                assert len(sents) > 200     # this will mess-up scoring
                continue
            sents.append(sent)
            gstrings.append(graph)
            if max_entries and len(gstrings) >= max_entries:
                break
        if max_entries and len(gstrings) >= max_entries:
            break

    # Save the data
    assert len(sents) == len(gstrings)
    print('Saving %d entries' % len(sents))
    write_lines(out_dir, 'sents.txt',    sents)
    write_lines(out_dir, 'gstrings.txt', gstrings)
