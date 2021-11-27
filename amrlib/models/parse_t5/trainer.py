import warnings
warnings.simplefilter('ignore')
import os
import logging
import torch
from   torch.utils.data import Dataset
from   transformers import T5ForConditionalGeneration, T5Tokenizer, set_seed
from   transformers import TrainingArguments, DataCollatorForSeq2Seq
from   .amr_t5_trainer import AMRT5Trainer
from   .penman_serializer import load_and_serialize

logger = logging.getLogger(__name__)


# Note that for save_steps, steps means gradient updates (not batch) so if
# gradient_accumulation_steps=4 and save_steps=1000, then checkpoint is saved every 4000 batches.
class Trainer(object):
    def __init__(self, args):
        # General arguments
        self.gen_args           = args['gen_args']
        self.model_name_or_path = self.gen_args['model_name_or_path']
        self.tok_name_or_path   = self.gen_args.get('tok_name_or_path', self.model_name_or_path)
        self.corpus_dir         = self.gen_args['corpus_dir']
        self.train_fn           = self.gen_args['train_fn']
        self.eval_fn            = self.gen_args.get('eval_fn')
        self.max_in_len         = self.gen_args['max_in_len']
        self.max_out_len        = self.gen_args['max_out_len']
        self.training_args      = TrainingArguments(**args['hf_args'])
        set_seed(self.training_args.seed)

    def train(self):
        # Create the output directory if needed
        os.makedirs(self.training_args.output_dir, exist_ok=True)
        # Load pretrained model and tokenizer
        print('Loading model and tokenizer')
        self.tokenizer = T5Tokenizer.from_pretrained(self.tok_name_or_path)
        self.model     = T5ForConditionalGeneration.from_pretrained(self.model_name_or_path)
        # Clear out the "task_specific_params" and add this one
        self.model.config.task_specific_params = {'parse_amr':self.gen_args}
        # Save the tokenizer
        if self.gen_args.get('save_tokenizer', False):
            self.tokenizer.save_pretrained(self.training_args.output_dir)
        # Load the datasets
        print('Building datasets')
        train_file_path = os.path.join(self.corpus_dir, self.train_fn)
        train_dataset   = self.build_dataset(train_file_path)
        print('Training data is {:,} after removing {:,} long entries'.format( \
            len(train_dataset), len(train_dataset.bad_indexes)))
        # Load the evaluation dataset
        if self.eval_fn:
            eval_file_path = os.path.join(self.corpus_dir, self.eval_fn)
            eval_samples   = load_and_serialize(eval_file_path)
            print('Evaluation data is {:,} samples'.format(len(eval_samples['graphs'])))
        else:
            eval_samples = None
        # Train the model
        print('Training')
        collator = DataCollatorForSeq2Seq(self.tokenizer, self.model, padding=True, max_length=self.max_out_len)
        trainer = AMRT5Trainer(model=self.model, args=self.training_args, train_dataset=train_dataset,
                    data_collator=collator, eval_tokenizer=self.tokenizer, eval_samples=eval_samples)
        # If resume_from_checkpoint is True it will look for the last checkpoint in the value of output_dir
        # passed via TrainingArguments.  If it's a path to a specific checkpoint it will use that saved
        # checkpoint folder to resume the training from.
        trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)
        # Save the results
        if self.gen_args.get('save_at_end', False):
            print('Saving model')
            trainer.save_model(self.training_args.output_dir)

    # Convert the AMR graphs into tokenized sentences
    def build_dataset(self, fpath):
        # Load the raw data
        entries = load_and_serialize(fpath) # returns a dict of lists
        # Tokenize the target in order to strip off any training samples that are too long.
        # Set return_overflowing_tokens=True to return overflowing_tokens', 'num_truncated_tokens'
        print('Tokenizing')
        in_enc  = self.tokenizer(entries['sents'],   padding=False, truncation=True, max_length=self.max_in_len,
                        return_overflowing_tokens=True)
        tgt_enc = self.tokenizer(entries['serials'], padding=False, truncation=True, max_length=self.max_out_len,
                        return_overflowing_tokens=True)
        # Identify any truncated data
        bi = set()
        for i, (ie, te) in enumerate(zip(in_enc['num_truncated_tokens'], tgt_enc['num_truncated_tokens'])):
            if ie > 0 or te > 0:
                bi.add( i )
        # Compile the output encodings, stripped of bad indexes
        # These will be passed directly to the model so make sure all the keys are correct, with no extras
        encodings = {}
        encodings['input_ids']      = [ie for i, ie in enumerate(in_enc['input_ids'])      if i not in bi]
        encodings['attention_mask'] = [ie for i, ie in enumerate(in_enc['attention_mask']) if i not in bi]
        encodings['labels']         = [te for i, te in enumerate(tgt_enc['input_ids'])     if i not in bi]
        return AMRDataset(encodings, bi)


# Torch DataSet used for feeding data to the training routine
class AMRDataset(Dataset):
    def __init__(self, encodings, bad_indexes):
        self.encodings      = encodings
        self.bad_indexes    = bad_indexes

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {k:v[idx] for k, v in self.encodings.items()}
