import os

# Simple container for holding process data
# This is a little complicated because preprocess creates data that is used by the model for input
# and other data that is used during postprocessing.  Keep track of it all here.
# Saving and loading is to facilitate training scripts.  During inference, data will be held internally.
class ProcData(object):
    def __init__(self, eng_lines=None, amr_lines=None,
                eng_tok_origpos_lines=None, amr_tuple_lines=None,
                eng_preproc_lines=None, amr_preproc_lines=None):
        self.eng_lines              = eng_lines
        self.amr_lines              = amr_lines
        self.eng_tok_origpos_lines  = eng_tok_origpos_lines
        self.amr_tuple_lines        = amr_tuple_lines
        self.eng_preproc_lines      = eng_preproc_lines
        self.amr_preproc_lines      = amr_preproc_lines

    # Save the preprocess and model input data (optionally the original x_lines data)
    def save(self, wk_dir, save_input_data=False, **kwargs):
        self.build_filenames(wk_dir, **kwargs)
        if save_input_data:
            self.save_lines(self.eng_fn, self.eng_lines)
            self.save_lines(self.amr_fn, self.amr_lines)
        self.save_lines(self.eng_tok_pos_fn, self.eng_tok_origpos_lines)
        self.save_lines(self.amr_tuple_fn,   self.amr_tuple_lines)
        with open(self.fa_in_fn, 'w') as f:
            for en_line, amr_line in zip(self.eng_preproc_lines, self.amr_preproc_lines):
                f.write('%s ||| %s\n' % (en_line, amr_line))

    # load data (not including the _preproc_lines)
    @classmethod
    def from_directory(cls, wk_dir, **kwargs):
        self = cls()
        self.build_filenames(wk_dir, **kwargs)
        self.eng_lines              = self.load_lines(self.eng_fn)
        self.amr_lines              = self.load_lines(self.amr_fn)
        self.eng_tok_origpos_lines  = self.load_lines(self.eng_tok_pos_fn)
        self.amr_tuple_lines        = self.load_lines(self.amr_tuple_fn)
        self.model_out_lines        = self.load_lines(self.model_out_fn)
        return self

    # Create default filenames as members
    def build_filenames(self, wk_dir, **kwargs):
        self.eng_fn         = os.path.join(wk_dir, kwargs.get('eng_fn',         'sents.txt'))
        self.amr_fn         = os.path.join(wk_dir, kwargs.get('amr_fn',         'gstrings.txt'))
        self.eng_tok_pos_fn = os.path.join(wk_dir, kwargs.get('eng_tok_pos_fn', 'eng_tok_origpos.txt'))
        self.amr_tuple_fn   = os.path.join(wk_dir, kwargs.get('amr_tuple_fn',   'amr_tuple.txt'))
        self.fa_in_fn       = os.path.join(wk_dir, kwargs.get('fa_in_fn',       'fa_in.txt'))
        self.model_out_fn   = os.path.join(wk_dir, kwargs.get('model_out_fn',   'model_out.txt'))

    # Save a list of lines to a file
    @staticmethod
    def save_lines(fn, lines):
        with open(fn, 'w') as f:
            for line in lines:
                f.write(line + '\n')

    # Load a list of lines from a file
    @staticmethod
    def load_lines(fn):
        with open(fn) as f:
            lines = [l.strip() for l in f]
        return lines
