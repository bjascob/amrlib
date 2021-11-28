import os
import logging
import torch
from   torch.utils.data import DataLoader
from   tqdm import tqdm
import transformers
from   transformers import Trainer as HFTrainer
from   transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from   ...evaluate.smatch_enhanced import get_entries, compute_smatch
from   .penman_serializer import PenmanDeSerializer
from   .inference import Inference


logger = logging.getLogger(__name__)


# Subclass the Huggingface Trainer to override the evaluate method
class AMRT5Trainer(HFTrainer):
    def __init__(self, model=None, args=None, data_collator=None, train_dataset=None, eval_dataset=None,
                    tokenizer=None, model_init=None, compute_metrics=None, callbacks=None,
                    optimizers=(None, None), **kwargs):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init,
                compute_metrics, callbacks, optimizers)
        self.gen_args       = self.model.config.task_specific_params['parse_amr']
        self.eval_tokenizer = kwargs['eval_tokenizer']  # passing "tokenizer" to trainer might cause issues
        self.eval_samples   = kwargs['eval_samples']

    # Compute smatch and return
    def evaluate(self, dataset=None, ignore_keys=None, metric_key_prefix='eval'):
        if self.eval_samples is None:
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
        print('Saving %s and %s' % (gold_out_fpath, pred_out_fpath))
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
        return {'smatch_f_score':f_score}
