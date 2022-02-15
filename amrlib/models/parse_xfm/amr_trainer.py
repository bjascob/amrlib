import os
import shutil
import logging
from   pathlib import Path
from   glob import glob
import torch
from   torch.utils.data import DataLoader
from   tqdm import tqdm
import transformers
from   transformers import Trainer as HFTrainer
from   transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from   transformers.trainer import TRAINER_STATE_NAME
from   ...evaluate.smatch_enhanced import get_entries, compute_smatch
from   .penman_serializer import PenmanDeSerializer
from   .inference import Inference

logger = logging.getLogger(__name__)


# Subclass the Huggingface Trainer to override the evaluate method
class AMRTrainer(HFTrainer):
    def __init__(self, model=None, args=None, data_collator=None, train_dataset=None, eval_dataset=None,
                    tokenizer=None, model_init=None, compute_metrics=None, callbacks=None,
                    optimizers=(None, None), **kwargs):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init,
                compute_metrics, callbacks, optimizers)
        self.gen_args       = self.model.config.task_specific_params['parse_amr']
        self.eval_tokenizer = kwargs['eval_tokenizer']  # passing "tokenizer" to trainer causing behavior changes
        self.eval_samples   = kwargs['eval_samples']

    # Compute smatch and return
    def evaluate(self, dataset=None, ignore_keys=None, metric_key_prefix='eval'):
        if self.eval_samples is None:
            return {}
        # Skip the evaluation on the first N epochs
        # self.state.epoch is a float so subract a little off in-case it's at 0.99 instead of 1.0
        first_eval_epoch = self.gen_args.get('first_eval_epoch', 0)
        if self.state.epoch < (first_eval_epoch - 0.1):
            print('Skipping evaluation')
            return {}
        print('Predicting AMRs and smatch scoring')
        # Setup the paths
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        checkpoint_dir    = os.path.join(self.args.output_dir, checkpoint_folder)
        os.makedirs(checkpoint_dir, exist_ok=True)  # model checkpoint save happens after eval
        gold_out_fpath     = os.path.join(checkpoint_dir, 'dev-gold.txt')
        pred_out_fpath     = os.path.join(checkpoint_dir, 'dev-pred.txt')
        # Separate the evaluation sample components
        ref_graphs = self.eval_samples['graphs']
        ref_sents  = self.eval_samples['sents']
        # Run evaluation
        inference = Inference(model=self.model, tokenizer=self.eval_tokenizer,
                                config=self.gen_args,
                                batch_size=self.gen_args.get('eval_batch_size', 12),
                                num_beams=self.gen_args.get('eval_num_beams', 1))
        gen_graphs = inference.parse_sents(ref_sents, disable_progress=False)
        assert len(gen_graphs) == len(ref_graphs)
        # Save the reference and generated graphs, omitting any that are None
        f_ref = open(gold_out_fpath, 'w')
        f_gen = open(pred_out_fpath, 'w')
        skipped = 0
        for ref_graph, gen_graph in zip(ref_graphs, gen_graphs):
            if gen_graph is None:
                skipped += 1
                continue
            f_ref.write(ref_graph + '\n\n')
            f_gen.write(gen_graph + '\n\n')
        f_ref.close()
        f_gen.close()
        print('Out of %d graphs, skipped %d that did not deserialize properly.' % (len(ref_graphs), skipped))
        # Run smatch evaluation.
        # Technically would not need to reload from the file but the smatch loader does some cleaning
        # so use `get_entries` just to be sure it's correct or at least consistant.
        gold_entries = get_entries(gold_out_fpath)
        test_entries = get_entries(pred_out_fpath)
        try:
            precision, recall, f_score = compute_smatch(test_entries, gold_entries)
        except:
            logger.exception('compute_smatch failed')
            precision, recall, f_score = 0, 0, 0
        print('SMATCH -> P: %.3f,  R: %.3f,  F: %.3f' % (precision, recall, f_score))
        f_score = round(f_score, 6)
        self.log({'smatch_f_score':f_score})
        return {'smatch_f_score':f_score}

    # Custom checkpoint saving. This will only save the best checkpoint, based on smatch score from evaluate().
    # Use the gen_args key "custom_save_checkpoint" to enable this.
    def _save_checkpoint(self, model, trial, metrics=None):
        custom_save = self.gen_args.get('custom_save_checkpoint', None)
        smatch_val  = None if metrics is None else metrics.get('smatch_f_score', None)
        if custom_save and smatch_val is None:
            logger.warning('Custom Save requested but no smatch_f_score in metrics. Reverting to default save')
        if not custom_save or smatch_val is None:
            return super()._save_checkpoint(model, trial, metrics)
        # Store the number of floating-point operations that went into the model
        self.store_flos()
        # Determine the new best metric / best model checkpoint
        if self.state.best_metric is None or smatch_val > self.state.best_metric:
            # Create the new checkpoint folder
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            new_chk_fpath = os.path.join(self.args.output_dir, checkpoint_folder)
            os.makedirs(new_chk_fpath, exist_ok=True)   # also created in evaluate()
            # Set internal state values
            self.state.best_metric           = smatch_val
            self.state.best_model_checkpoint = new_chk_fpath
            # Save the model and trainer state
            self.save_model(new_chk_fpath, _internal_call=True)
            self.state.save_to_json(os.path.join(new_chk_fpath, TRAINER_STATE_NAME))
        else:
            print(f"Checkpoint not saved. Smatch score lower than best.")
        # Delete older checkpoints (might be empty except for evaluation data)
        checkpoint_list = [str(x) for x in Path(self.args.output_dir).glob(f"{PREFIX_CHECKPOINT_DIR}-*")]
        for checkpoint in checkpoint_list:
            if checkpoint != self.state.best_model_checkpoint:
                shutil.rmtree(checkpoint)
