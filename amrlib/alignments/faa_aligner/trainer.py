import os
import json
import subprocess


class Trainer:
    def __init__(self, in_fpath, out_dir, **kwargs):
        # If the bin_dir is not provided, get it from the environment, but default to ''
        # which means it must be in the path
        bin_dir         = os.environ.get('FABIN_DIR', '')
        bin_dir         = kwargs.get('bin_dir', bin_dir)
        self.fwd_out    = os.path.join(out_dir, 'fa_out_fwd.txt')
        self.rev_out    = os.path.join(out_dir, 'fa_out_rev.txt')
        self.final_out  = os.path.join(out_dir, 'model_out.txt')
        self.fwd_params = os.path.join(out_dir, 'fwd_params')
        self.rev_params = os.path.join(out_dir, 'rev_params')
        self.fwd_err    = os.path.join(out_dir, 'fwd_err')
        self.rev_err    = os.path.join(out_dir, 'rev_err')
        self.param_fn   = os.path.join(out_dir, 'train_params.json')
        fast_align      = os.path.join(bin_dir, 'fast_align')
        atools          = os.path.join(bin_dir, 'atools')
        p               = {}
        p['I']          = kwargs.get('I', 5)
        p['q']          = kwargs.get('q', 0.08)
        p['a']          = kwargs.get('a', 0.01)
        p['T']          = kwargs.get('T', 4)
        p['heuristic']  = kwargs.get('heuristic', 'grow-diag-final-and')
        self.params     = p
        # Create the actual commands to execute
        fwd_cmd   = '%s -i %s -d -o -v -I %d -q %f -a %f -T %d -p %s' % \
                    (fast_align, in_fpath, p['I'], p['q'], p['a'], p['T'], self.fwd_params)
        rev_cmd   = '%s -i %s -d -o -v -I %d -q %f -a %f -T %d -p %s -r' % \
                    (fast_align, in_fpath, p['I'], p['q'], p['a'], p['T'], self.rev_params)
        tools_cmd = '%s -i %s -j %s -c %s' % (atools, self.fwd_out, self.rev_out, p['heuristic'])
        self.fwd_cmd   = fwd_cmd.split()
        self.rev_cmd   = rev_cmd.split()
        self.tools_cmd = tools_cmd.split()

    def train(self):
        # Run the training
        with open(self.fwd_out, 'w') as fout, open(self.fwd_err, 'w') as ferr:
            subprocess.run(self.fwd_cmd, stdout=fout, stderr=ferr)
        with open(self.rev_out, 'w') as fout, open(self.rev_err, 'w') as ferr:
            subprocess.run(self.rev_cmd, stdout=fout, stderr=ferr)
        with open(self.final_out, 'w') as fout:
            subprocess.run(self.tools_cmd, stdout=fout)
        # Get a few final parameters
        self.params['fwd_T'], self.params['fwd_m'] = self.read_err(self.fwd_err)
        self.params['rev_T'], self.params['rev_m'] = self.read_err(self.rev_err)
        # Save the parameters for use during inference
        with open(self.param_fn, 'w') as f:
            json.dump(self.params, f, indent=4)

    def read_err(self, err):
        T, m = '', ''
        with open(err) as f:
            for line in f:
                # expected target length = source length * N
                if 'expected target length' in line:
                    m = line.split()[-1]
                # final tension: N
                elif 'final tension' in line:
                    T = line.split()[-1]
        return float(T), float(m)
