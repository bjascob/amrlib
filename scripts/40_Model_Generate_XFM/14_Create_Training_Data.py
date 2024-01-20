#!/usr/bin/env python3
import setup_run_dir    # this import tricks script to run from 2 levels up
import os
from   random import shuffle
import logging
from   tqdm import tqdm
from   amrlib.graph_processing.amr_loading import load_amr_entries
from   amrlib.utils.logging import setup_logging, silence_penman, WARN
from   amrlib.models.generate_xfm.model_input_helper import ModelInputHelper


logger = logging.getLogger(__name__)

# Nomenclature
# xx.features               xx with fields added to the metadata for the spacy parse
# xx.features.nowiki        above with wiki tags removed from the graphs
# xx.features.nowiki.tdata  the standard AMR data plus a copy with tense tags added, shuffled together
#
# Take graphs that are annotated (tokens, pos, ...) and align them then tag the graphs.
# Save files with the tagged and untagged data together in a single training file
if __name__ == '__main__':
    setup_logging(level=WARN, logfname='logs/create_td_gen_xfm.log')
    silence_penman()
    data_dir = 'amrlib/data/tdata_generate_xfm'
    base_fns = ('train.txt',)

    # Loop through the files
    for base_fn in base_fns:
        infn = os.path.join(data_dir, base_fn + '.features.nowiki')
        print('Loading and processing', infn)
        entries = load_amr_entries(infn)
        tagged_entries = []
        for entry in tqdm(entries, ncols=100):
            tagged_entry = ModelInputHelper(entry).get_tagged_with_meta()
            tagged_entries.append(tagged_entry)
        # Save the tagged and untagged entries into a single file, shuffled together
        all_entries = entries + tagged_entries
        shuffle(all_entries)
        outfn = infn + '.tdata'
        print('Saving to', outfn)
        with open(outfn, 'w') as f:
            for entry in all_entries:
                f.write(entry + '\n\n')
