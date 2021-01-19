#!/usr/bin/python3
import os
import re
from   .process_utils import stem_4_letters_word, stem_4_letters_line, stem_4_letters_string
from   .process_utils import filter_eng_by_stopwords, get_lineartok_with_rel
from   .process_utils import get_id_mapping_uniq


# Set the default data data for misc files
default_res_dir = os.path.dirname(os.path.realpath(__file__))
default_res_dir = os.path.realpath(os.path.join(default_res_dir, 'resources'))


# Preprocess for inference
def preprocess_infer(wk_dir, eng_lines, amr_lines, **kwargs):
    assert len(eng_lines) == len(amr_lines)
    # Resource filenames
    res_dir         = kwargs.get('res_dir', default_res_dir)
    eng_sw_fn       = kwargs.get('eng_sw_fn', os.path.join(res_dir, 'eng_stopwords.txt'))
    amr_sw_fn       = kwargs.get('amr_sw_fn', os.path.join(res_dir, 'amr_stopwords.txt'))
    # Output filenames
    eng_tok_pos_fn  = os.path.join(wk_dir, kwargs.get('eng_tok_pos_fn', 'eng_tok_origpos.txt'))
    amr_tuple_fn    = os.path.join(wk_dir, kwargs.get('amr_tuple_fn', 'amr_tuple.txt'))

    # Filter out stopwords from sentences
    eng_tok_filtered_lines, eng_tok_origpos_lines = filter_eng_by_stopwords(eng_lines, eng_sw_fn)

    # Save for post-processing
    with open(eng_tok_pos_fn, 'w') as f:
        for i, l in enumerate(eng_tok_origpos_lines):
            assert l.strip(), '!!! ERROR Empty line# %d. This will cause issues and must be fixed !!!' % i
            f.write(l + '\n')

    # Stem sentence tokens
    eng_tok_stemmed_lines = [stem_4_letters_line(l) for l in eng_tok_filtered_lines]

    # Process the AMR data / remove stopwords
    amr_linear_lines, amr_tuple_lines = get_lineartok_with_rel(amr_lines, amr_sw_fn)

    # Save for post-processing
    with open(amr_tuple_fn, 'w') as f:
        for l in amr_tuple_lines:
            f.write(l + '\n')

    # Stem the AMR lines
    amr_linear_stemmed_lines = []
    for line in amr_linear_lines:
        new_tokens = []
        for token in line.split():
            token = re.sub(r'\-[0-9]{2,3}$', '', token)
            token = token.replace('"', '')
            token = stem_4_letters_word(token).strip()
            new_tokens.append(token)
        amr_linear_stemmed_lines.append(' '.join(new_tokens))

    return eng_tok_stemmed_lines, amr_linear_stemmed_lines


# Preprocess the training data.  This is the similar to inference but add a lot of
# extra translation lines from resource files, etc..
def preprocess_train(wk_dir, eng_lines, amr_lines, **kwargs):
    repeat_td       = kwargs.get('repeat_td', 10)   # 10X is original value from isi aligner
    # Resource filenames
    res_dir         = kwargs.get('res_dir', default_res_dir)
    prep_roles_fn   = kwargs.get('prep_roles_fn', os.path.join(res_dir, 'prep-roles_id.txt'))
    eng_id_map_fn   = kwargs.get('eng_id_map_fn', os.path.join(res_dir, 'eng_id_map.txt'))
    amr_id_map_fn   = kwargs.get('amr_id_map_fn', os.path.join(res_dir, 'amr_id_map.txt'))

    # Run the inference process which creates the basic translation data
    eng_tok_stemmed_lines, amr_linear_stemmed_lines = preprocess_infer(wk_dir, eng_lines, amr_lines, **kwargs)

    # Get tokens common between the two datasets (obvious translations
    common_tok_lines = get_id_mapping_uniq(eng_tok_stemmed_lines, amr_linear_stemmed_lines)
    eng_td_lines = common_tok_lines[:]  # copy

    # Append the second field in prep-roles.id.txt
    res_fn = os.path.join(res_dir, 'prep-roles_id.txt')
    with open(res_fn) as f:
        prep_roles_lines = [l.strip() for l in f]
    add_lines = [x.split()[1] for x in prep_roles_lines]
    add_lines = [stem_4_letters_line(l) for l in add_lines]
    eng_td_lines = eng_td_lines + add_lines

    # Append some custom translations data
    with open(eng_id_map_fn) as f:
        add_lines = [stem_4_letters_line(l.strip()) for l in f]
    eng_td_lines = eng_td_lines + add_lines

    # Get a copy of the original common_tok_lines data
    amr_td_lines = common_tok_lines[:]

    # Append the first field in prep-roles.id.txt
    add_lines = [x.split()[0] for x in prep_roles_lines]    # loaded above
    add_lines = [stem_4_letters_line(l) for l in add_lines]
    amr_td_lines = amr_td_lines + add_lines

    # Append some custom translation data
    with open(amr_id_map_fn) as f:
        add_lines = [stem_4_letters_line(l.strip()) for l in f]
    amr_td_lines = amr_td_lines + add_lines

    # Create the final training data using the original sentences
    # and 10X copies of the additional data (other translations)
    eng_td_lines = eng_tok_stemmed_lines    + [l for _ in range(repeat_td) for l in eng_td_lines]
    amr_td_lines = amr_linear_stemmed_lines + [l for _ in range(repeat_td) for l in amr_td_lines]
    assert len(eng_td_lines) == len(amr_td_lines)

    return eng_td_lines, amr_td_lines
