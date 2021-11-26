import os
import json
import logging
import penman
from   penman.models.amr import model as amr_model
import torch
from   torch.optim import AdamW
from   torch.cuda.amp import autocast
from   torch.cuda.amp.grad_scaler import GradScaler
import transformers
from   tqdm import tqdm
from   ...evaluate.smatch_enhanced import get_entries, compute_smatch
from   .model_utils import instantiate_model_and_tokenizer, get_dataloader, load_state_dict_from_checkpoint
from   .amr_rw import read_raw_amr_data
from   .inference import Inference

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config, device='cuda:0'):
        self.config        = config
        self.device        = torch.device(device)
        self.model_dir     = self.config['model_dir']
        self.dev_gold_path = os.path.join(self.model_dir, 'dev-gold.txt')
        self.dev_pred_path = os.path.join(self.model_dir, 'dev-pred.txt')
        self.best_smatch   = -1     # May get reloaded if using a checkpoint
        self.start_epoch   = 1      # May get reloaded if using a checkpoint
        os.makedirs(self.model_dir, exist_ok=True)

    def train(self, checkpoint=None):
        self.load_model(checkpoint) # sets self.model, tokenizer, optimizer, .., best_smatch, start_epoch
        self.load_training_data()   # sets self.train_loader
        self.load_eval_data()       # sets self.inference, graphs_gold
        # Loop through max epochs
        assert self.start_epoch < self.config['max_epochs']     # May fail if reloading a checkpoint
        for epoch in range(self.start_epoch, self.config['max_epochs']+1):
            # Setup batch
            print('Training epoch %d' % epoch)
            trn_amr_loss = RunningAverage()
            self.optimizer.zero_grad()
            pbar = tqdm(total=len(self.train_loader.dataset), ncols=100)
            self.set_pbar_desc_train(pbar, None)
            self.model.train()
            # Loop through all the data
            for bnum, batch in enumerate(self.train_loader):
                x, y, extra = batch
                with autocast(enabled=self.config['fp16']):
                    rdict = self.model(**x, **y)
                    loss = rdict['loss']
                self.scaler.scale((loss / self.config['accum_steps'])).backward()
                trn_amr_loss.update(loss.item())
                # Perform an update every accum_steps
                if (bnum+1) % self.config['accum_steps']==0:
                    self.step_otimizer()
                # Update progress
                pbar.update(x['input_ids'].shape[0])
                self.set_pbar_desc_train(pbar, trn_amr_loss.value)
            pbar.close()
            # Perform an update with the last batch if it wasn't already done in the loop
            if (bnum+1) % self.config['accum_steps'] != 0:
                self.step_otimizer()
            # Run evaluate, compute smatch and save the model if it's the new best
            try:
                smatch = self.evaluate()
                if smatch > self.best_smatch:
                    self.best_smatch = smatch
                    self.save_and_remove_checkpoints(epoch, smatch)
            except:
                print('!! Evaluation / save failed !!')
                logger.exception('Evaluation or model save failed')
            print()

    # Run Inference and evaluate the model
    def evaluate(self):
        self.model.eval()
        sents = [g.metadata['snt'] for g in self.graphs_gold]
        graphs_gen = self.inference.parse_sents(sents, return_penman=True, disable_progress=False,
                        pbar_desc='%-14s' % 'Evaluating:')
        assert len(graphs_gen) == len(self.graphs_gold)
        # Detect bad graphs. In Penman 1.2.0, metadata does not impact penam.Graph.__eq__()
        num_bad = sum(g == Inference.invalid_graph for g in graphs_gen)
        print('Out of %d graphs, %d did not generate properly.' % (len(graphs_gen), num_bad))
        # Save the final graphs
        print('Generated graphs written to', self.dev_pred_path)
        penman.dump(graphs_gen, self.dev_pred_path, indent=6, model=amr_model)
        # Run smatch
        try:
            gold_entries = get_entries(self.dev_gold_path)
            test_entries = get_entries(self.dev_pred_path)
            precision, recall, f_score = compute_smatch(test_entries, gold_entries)
            print('SMATCH -> P: %.3f,  R: %.3f,  F: %.3f' % (precision, recall, f_score))
        except:
            logger.exception('Failed to compute smatch score.')
            precision, recall, f_score = 0, 0, 0
        return f_score

    # Save the checkpoints if this is the best score
    def save_and_remove_checkpoints(self, epoch, smatch):
        prev_checkpoints = [fn for fn in os.listdir(self.model_dir) if fn.endswith('.pt')]
        model_fn = 'checkpoint_epoch_%02d_smatch_%04d.pt' % (epoch, smatch*10000)
        model_fpath = os.path.join(self.model_dir, model_fn)
        # Create the dictionary with the optional optimizer and save it
        print('Saving new, best model to', model_fpath)
        save_dict = {'model': self.model.state_dict()}
        if self.config.get('save_optimizer'):
            save_dict['optimizer'] = self.optimizer.state_dict()
            save_dict['scheduler'] = self.scheduler.state_dict()
        torch.save(save_dict, model_fpath)
        # Save the config file
        self.config['smatch_dev'] = smatch
        self.config['last_epoch'] = epoch
        with open(os.path.join(self.model_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)
        # Remove previous checkpoints
        for chkpt_fn in prev_checkpoints:
            os.remove(os.path.join(self.model_dir, chkpt_fn))

    # Load and setup the model, tokenizer, optimizer, etc..
    def load_model(self, checkpoint=None):
        print('Loading model from', self.config['model'])
        self.model, self.tokenizer = instantiate_model_and_tokenizer(self.config['model'],
                                        additional_tokens_smart_init=self.config['smart_init'],
                                        dropout=self.config['dropout'],
                                        attention_dropout=self.config['attention_dropout'],
                                        penman_linearization=self.config['penman_linearization'],
                                        collapse_name_ops=self.config['collapse_name_ops'],
                                        use_pointer_tokens=self.config['use_pointer_tokens'],
                                        raw_graph=self.config['raw_graph'])
        self.model.to(self.device)
        # Load optimization components
        self.optimizer = AdamW(self.model.parameters(),  lr=self.config['learning_rate'],
                                weight_decay=self.config['weight_decay'])
        self.scheduler = transformers.get_constant_schedule_with_warmup(self.optimizer,
                                num_warmup_steps=self.config['warmup_steps'])
        self.scaler = GradScaler(enabled=self.config['fp16'])
        # Reload checkpoint model weights and optimizer params if loading from a checkpoint
        if checkpoint is not None:
            print('Checkpoint %s restored' % checkpoint)
            load_state_dict_from_checkpoint(checkpoint, self.model, self.optimizer, self.scheduler)
            # Try to load the smatch score and last_epoch from the config in the model directory.
            try:
                with open(os.path.join(self.model_dir, 'config.json')) as f:
                    model_config = json.load(f)
                self.best_smatch = model_config['smatch_dev']
                self.start_epoch = model_config['last_epoch'] + 1
            except:
                logger.exception('Unable to load config file in model directory')

    # Setup the training data loader
    def load_training_data(self):
        print('Loading train data from', self.config['train'])
        self.train_loader = get_dataloader(self.tokenizer,
                                glob_pattern=self.config['train'],
                                evaluation=False,
                                batch_size=self.config['batch_size'],
                                use_recategorization=self.config['use_recategorization'],
                                remove_longer_than=self.config['remove_longer_than'],
                                remove_wiki=self.config['remove_wiki'],
                                dereify=self.config['dereify'],
                                device=self.device)

    # Setup the inference object and create the gold data test file
    def load_eval_data(self):
        print('Loading eval data from ', self.config['dev'])
        self.inference = Inference(model=self.model, tokenizer=self.tokenizer, device=self.device,
                                    num_beams=self.config['eval_beam_size'],
                                    batch_size=self.config['eval_batch_sents'],
                                    config=self.config)
        self.graphs_gold = read_raw_amr_data(self.config['dev'],
                            use_recategorization=self.config['use_recategorization'],
                            dereify=self.config['dereify'],
                            remove_wiki=self.config['remove_wiki'])
        penman.dump(self.graphs_gold, self.dev_gold_path, indent=6, model=amr_model)

    # Function to update the model's parameters for accumulated loss
    def step_otimizer(self):
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_norm'])
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self.scheduler.step()

    # Update tqdm progress bar description with loss values
    @staticmethod
    def set_pbar_desc_train(pbar, av_loss):
        desc = 'Loss: '
        if av_loss is None:
            desc += ' '*8
        else:
            desc += '%8.3f' % av_loss
        pbar.set_description(desc)


# Same as the pytorch-ignite running-average computation
class RunningAverage(object):
    def __init__(self, alpha=0.98):
        self.alpha = alpha
        self.value = None
    def update(self, new_val):
        if self.value is None:
            self.value = new_val
        else:
            self.value = self.value*self.alpha + (1.0-self.alpha)*new_val
