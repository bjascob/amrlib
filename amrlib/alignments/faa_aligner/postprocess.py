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
def postprocess(data, **kwargs):
    # Read the output of the aligner or use the supplied input above
    # fast_align outputs with a dash but the code from the isi aligner is setup for spaces
    giza_align_lines = [l.strip().replace('-', ' ') for l in data.model_out_lines]
    isi_align_lines  = giza2isi(giza_align_lines)
    num_lines        = len(data.amr_lines)
    align_real_lines = swap(isi_align_lines)[:num_lines]  # rm data added for training, past original sentences

    # Align the original position lines
    align_origpos_lines = map_ibmpos_to_origpos_amr_as_f(data.eng_tok_origpos_lines, align_real_lines)

    # Get the aligned tuples
    aligned_tuple_lines = get_aligned_tuple_amr_as_f_add_align(data.amr_tuple_lines, align_origpos_lines)

    # Create amr graphs with surface alignments
    amr_surface_aligns = feat2tree.align(data.amr_lines, aligned_tuple_lines)
    assert len(amr_surface_aligns) == len(data.amr_lines)

    # Get the final alignment string from the surface alignments
    ga = GetAlignments.from_amr_strings(amr_surface_aligns)
    alignment_strings = ga.get_alignments()

    return amr_surface_aligns, alignment_strings
