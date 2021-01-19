import os
import json
import subprocess
import logging
import tarfile
from   .preprocess import preprocess_infer
from   .postprocess import postprocess
from   .get_alignments import GetAlignments
from   ..penman_utils import to_graph_line
from   ...defaults import data_dir

logger = logging.getLogger(__name__)

this_dir = os.path.dirname(os.path.realpath(__file__))


class FAA_Aligner(object):
    def __init__(self, **kwargs):
        self.model_dir    = kwargs.get('model_dir',    os.path.join(data_dir, 'model_aligner_faa'))
        self.working_dir  = kwargs.get('working_dir',  os.path.join(data_dir, 'working_faa_aligner'))
        self.model_tar_fn = kwargs.get('model_tar_fn', os.path.join(this_dir, 'model_aligner_faa.tar.gz'))
        os.makedirs(self.working_dir, exist_ok=True)
        self.setup_model_dir()
        self.aligner = TrainedAligner(self.model_dir, **kwargs)
        self.aligner.check_for_binaries()   # Will raise FileNotFoundError if binaries can't be found

    # check the model directory, if it doesn't have the metadata file try to create
    # the directory from the tar.gz file
    def setup_model_dir(self):
        # Check for the metadata and if so, consider the model ready to go
        if os.path.isfile(os.path.join(self.model_dir, 'amrlib_meta.json')):
            return True
        # if there's a local copy, etract it
        elif os.path.isfile(self.model_tar_fn):
            tar = tarfile.open(self.model_tar_fn)
            tar.extractall(path=data_dir)
            logger.info('Extracting a local copy of model')
        else:
            logger.critical('No model in model_dir and no local version available to extract')
            return False

    def align_sents(self, sents, gstrings):
        gstrings = [to_graph_line(g) for g in gstrings]
        eng_td_lines, amr_td_lines = preprocess_infer(self.working_dir, sents, gstrings)
        fa_out_lines = self.aligner.align(eng_td_lines, amr_td_lines)
        amr_surface_aligns, alignment_strings = postprocess(self.working_dir, fa_out_lines)
        return amr_surface_aligns, alignment_strings


# Code adapted from from https://github.com/clab/fast_align/blob/master/src/force_align.py
class TrainedAligner:
    def __init__(self, model_in_dir, **kwargs):
        # If the bin_dir is not provided, get it from the environment, but default
        # to '' which means it must be in the path
        bin_dir         = os.environ.get('FABIN_DIR', '')
        bin_dir         = kwargs.get('bin_dir', bin_dir)
        self.fast_align = os.path.join(bin_dir, 'fast_align')
        self.atools     = os.path.join(bin_dir, 'atools')
        fwd_params_fn   = os.path.join(model_in_dir, 'fwd_params')
        rev_params_fn   = os.path.join(model_in_dir, 'rev_params')
        # Get the parameters from the metadata
        with open(os.path.join(model_in_dir, 'amrlib_meta.json')) as f:
            meta = json.load(f)
        p = meta['train_params']
        # timeout the exe to exit
        self.timeout = kwargs.get('timeout', 1.0)
        # Create the actual commands to execute
        fwd_cmd   = '%s -i - -d -q %f -a %f -T %f -m %f -f %s' % \
                    (self.fast_align, p['q'], p['a'], p['fwd_T'], p['fwd_m'], fwd_params_fn)
        rev_cmd   = '%s -i - -d -q %f -a %f -T %f -m %f -f %s -r' % \
                    (self.fast_align, p['q'], p['a'], p['fwd_T'], p['fwd_m'], rev_params_fn)
        tools_cmd = '%s -i - -j - -c %s' % (self.atools, p['heuristic'])
        self.fwd_cmd   = fwd_cmd.split()
        self.rev_cmd   = rev_cmd.split()
        self.tools_cmd = tools_cmd.split()

    # Open a connection to the subprocess in text mode
    @staticmethod
    def popen_io(cmd):
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)

    def align(self, eng_td_lines, amr_td_lines):
        # Combine lines into fast align input format
        lines = ['%s ||| %s' % (el, al) for el, al in zip(eng_td_lines, amr_td_lines)]
        # Open connections to the alignment binaries
        self.fwd_align = self.popen_io(self.fwd_cmd)
        self.rev_align = self.popen_io(self.rev_cmd)
        self.tools     = self.popen_io(self.tools_cmd)
        # Input to fast_align
        fa_in = '\n'.join([l.strip() for l in lines])
        fwd_out, fwd_err = self.fwd_align.communicate(fa_in, timeout=self.timeout)
        rev_out, fwd_err = self.rev_align.communicate(fa_in, timeout=self.timeout)
        # output is     f words ||| e words ||| links ||| score
        fwd_lines = [l.split('|||')[2].strip() for l in fwd_out.splitlines() if l]
        rev_lines = [l.split('|||')[2].strip() for l in rev_out.splitlines() if l]
        # Input to atools
        at_in = '\n'.join(['%s\n%s' % (fl, rl) for fl, rl in zip(fwd_lines, rev_lines)])
        at_out, at_err = self.tools.communicate(at_in, timeout=self.timeout)
        at_lines = [l.strip() for l in at_out.splitlines()]
        return at_lines

    # This will raise FileNotFoundError if either call fails
    # Note that both commands trigger the help message and will produce a return-code of 1
    # which is typically considered and error
    def check_for_binaries(self):
        ret_fa   = subprocess.run(self.fast_align, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        ret_tool = subprocess.run(self.atools, stderr=subprocess.PIPE, stdout=subprocess.PIPE)