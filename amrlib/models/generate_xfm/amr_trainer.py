import os
import shutil
import logging
from   pathlib import Path
from   glob import glob
import torch
from   torch.utils.data import DataLoader
from   tqdm import tqdm
from   nltk.tokenize import word_tokenize
import transformers
from   transformers import Trainer as HFTrainer
from   transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from   transformers.trainer import TRAINER_STATE_NAME
from   ...evaluate.bleu_scorer import BLEUScorer
from   .inference import Inference

logger = logging.getLogger(__name__)


def save_sents(fpath, sents):
    with open(fpath, 'w') as f:
        for sent in sents:
            f.write(sent + '\n')

def tokenize_and_lower(sents):
    sents = [word_tokenize(s.strip().lower()) for s in sents]
    return sents

# Get the sentence from an AMR graph string
def get_sentence(graph):
    for line in graph.splitlines():
        if line.startswith('# ::snt'):
            return line[len('# :snt')+1:].strip()
    logger.error('Evaluation data contains no meta-date for sentences')


# Subclass the Huggingface Trainer to override the evaluate method
class AMRTrainer(HFTrainer):
    def __init__(self, model=None, args=None, data_collator=None, train_dataset=None, eval_dataset=None,
                    tokenizer=None, model_init=None, compute_metrics=None, callbacks=None,
                    optimizers=(None, None), **kwargs):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init,
                compute_metrics, callbacks, optimizers)
        self.gen_args       = self.model.config.task_specific_params['translation_amr_to_text']
        self.eval_tokenizer = kwargs['eval_tokenizer']  # passing "tokenizer" to trainer causing behavior changes
        self.eval_samples   = kwargs['eval_samples']

    # Compute bleu and return
    def evaluate(self, dataset=None, ignore_keys=None, metric_key_prefix='eval'):
        if self.eval_samples is None:
            return {}
        # Skip the evaluation on the first N epochs
        # self.state.epoch is a float so subract a little off in-case it's at 0.99 instead of 1.0
        first_eval_epoch = self.gen_args.get('first_eval_epoch', 0)
        if self.state.epoch < (first_eval_epoch - 0.1):
            print('Skipping evaluation')
            return {}
        print('Predicting and BLEU scoring')
        # Setup the paths
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        checkpoint_dir    = os.path.join(self.args.output_dir, checkpoint_folder)
        os.makedirs(checkpoint_dir, exist_ok=True)  # model checkpoint save happens after eval
        # Setup the evaluation samples (these should contain the metadata)
        ref_graphs = self.eval_samples
        ref_sents  = [get_sentence(g) for g in ref_graphs]
        # Run evaluation
        inference = Inference(model=self.model, tokenizer=self.eval_tokenizer,
                              batch_size=self.gen_args.get('eval_batch_size', 8),
                              num_beams=self.gen_args.get('eval_num_beams', 1),  num_ret_seq=1)
        use_tense = self.gen_args.get('eval_use_tense', False)
        gen_sents, clips = inference.generate(ref_graphs, disable_progress=False, use_tense=use_tense)
        assert len(gen_sents) == len(ref_sents)
        # Filter out any clipped graphs as invalid tests (This will raise the BLEU score)
        if 1:
            print(f'{sum(clips)} graphs were clipped during tokenization and are removed from scoring')
            assert len(ref_sents) == len(gen_sents) == len(clips)
            ref_sents = [s for s, c in zip(ref_sents, clips) if not c]
            gen_sents = [a for a, c in zip(gen_sents, clips) if not c]
        else:
            print(f'{sum(clips)} graphs were clipped during tokenization and are included in the score')
        # Save the data
        ref_out_fpath = os.path.join(checkpoint_dir, 'reference_sents.txt')
        gen_out_fpath = os.path.join(checkpoint_dir, 'generated_sents.txt')
        save_sents(ref_out_fpath, ref_sents)
        save_sents(gen_out_fpath, gen_sents)
        # Modify sentences for BLEU scoring
        ref_sents = tokenize_and_lower(ref_sents)
        gen_sents = tokenize_and_lower(gen_sents)
        try:
            bleu_scorer = BLEUScorer()
            bleu_score, _, _ = bleu_scorer.compute_bleu(ref_sents, gen_sents)
        except:
            logger.exception('compute bleu score failed')
            bleu_score = 0
        print('BLEU score: %5.2f' % (bleu_score*100.))
        bleu_score = round(bleu_score, 6)
        self.log({'bleu_score':bleu_score})
        return {'bleu_score':bleu_score}

    # Custom checkpoint saving. This will only save the best checkpoint, based on bleu score from evaluate().
    # Use the gen_args key "custom_save_checkpoint" to enable this.
    def _save_checkpoint(self, model, trial, metrics=None):
        custom_save = self.gen_args.get('custom_save_checkpoint', None)
        bleu_val  = None if metrics is None else metrics.get('bleu_score', None)
        if custom_save and bleu_val is None:
            logger.warning('Custom Save requested but no bleu_score in metrics. Reverting to default save')
        if not custom_save or bleu_val is None:
            return super()._save_checkpoint(model, trial, metrics)
        # Store the number of floating-point operations that went into the model
        self.store_flos()
        # Determine the new best metric / best model checkpoint
        if self.state.best_metric is None or bleu_val > self.state.best_metric:
            # Create the new checkpoint folder
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            new_chk_fpath = os.path.join(self.args.output_dir, checkpoint_folder)
            os.makedirs(new_chk_fpath, exist_ok=True)   # also created in evaluate()
            # Set internal state values
            self.state.best_metric           = bleu_val
            self.state.best_model_checkpoint = new_chk_fpath
            # Save the model and trainer state
            self.save_model(new_chk_fpath, _internal_call=True)
            self.state.save_to_json(os.path.join(new_chk_fpath, TRAINER_STATE_NAME))
        else:
            print(f"Checkpoint not saved. bleu score lower than best.")
        # Delete older checkpoints (might be empty except for evaluation data)
        checkpoint_list = [str(x) for x in Path(self.args.output_dir).glob(f"{PREFIX_CHECKPOINT_DIR}-*")]
        for checkpoint in checkpoint_list:
            if checkpoint != self.state.best_model_checkpoint:
                shutil.rmtree(checkpoint)
