#!/usr/bin/python3
import os
import io
import sys
from   types import SimpleNamespace
from   .import feat2tree
from   .process_utils import map_ibmpos_to_origpos_amr_as_f, get_aligned_tuple_amr_as_f_add_align
from   .process_utils import giza2isi, swap, add_word_pos
from   .get_alignments import GetAlignments


# if model_out_lines is None, read from the file
def postprocess(wk_dir, model_out_lines=None, eng_lines=None, amr_lines=None, **kwargs):
    # Input filenames
    eng_fn          = os.path.join(wk_dir, kwargs.get('eng_fn', 'sents.txt'))
    amr_fn          = os.path.join(wk_dir, kwargs.get('amr_fn', 'gstrings.txt'))
    eng_tok_pos_fn  = os.path.join(wk_dir, kwargs.get('eng_tok_pos_fn', 'eng_tok_origpos.txt'))
    amr_tuple_fn    = os.path.join(wk_dir, kwargs.get('amr_tuple', 'amr_tuple.txt'))
    model_out_fn    = os.path.join(wk_dir, kwargs.get('model_out_fn', 'model_out.txt'))
    # Error log
    align_to_str_fn = os.path.join(wk_dir, kwargs.get('align_to_str_fn', 'align_to_str.err'))

    # Read the input files and get the number of lines, which must be the same
    if eng_lines is None or amr_lines is None:
        with open(eng_fn) as f:
            eng_lines = [l.strip() for l in f]
        with open(amr_fn) as f:
            amr_lines = [l.strip() for l in f]
        assert len(eng_lines) == len(amr_lines)
    lines_number = len(eng_lines)

    # Read the output of the aligner or use the supplied input above
    # fast_align outputs with a dash but the code from the isi aligner is setup for spaces
    if model_out_lines is None:
        with open(model_out_fn) as f:
            model_out_lines = f.readlines()
    # fast_align outputs with dashes, giza does this without
    giza_align_lines = [l.strip().replace('-', ' ') for l in model_out_lines]
    isi_align_lines  = giza2isi(giza_align_lines)
    align_real_lines = swap(isi_align_lines)[:lines_number]  # rm data added for training, past original sentences

    # Load the original sentence tokenization positions (created in pre-process)
    with open(eng_tok_pos_fn) as f:
        eng_tok_origpos_lines = [l.strip() for l in f]
    align_origpos_lines = map_ibmpos_to_origpos_amr_as_f(eng_tok_origpos_lines, align_real_lines)

    # Load the amr tuples from the pre-process steps and add the alignments
    with open(amr_tuple_fn) as f:
        amr_tuple_lines = [l.strip() for l in f]
    aligned_tuple_lines = get_aligned_tuple_amr_as_f_add_align(amr_tuple_lines, align_origpos_lines)

    # Create amr graphs with surface alignments
    amr_surface_aligns = feat2tree.align(amr_lines, aligned_tuple_lines, log_fn=align_to_str_fn)
    assert len(amr_surface_aligns) == len(eng_lines)

    # Get the final alignment string from the surface alignments
    ga = GetAlignments.from_amr_strings(amr_surface_aligns)
    alignment_strings = ga.get_alignments()

    return amr_surface_aligns, alignment_strings
