import warnings
warnings.simplefilter('ignore')
import os
import logging
import torch
from   torch.utils.data import Dataset
from   ...graph_processing.amr_loading import load_amr_graph_sent
from   transformers import T5ForConditionalGeneration, T5Tokenizer, set_seed
from   transformers import TrainingArguments
from   transformers import Trainer as T5Trainer

logger = logging.getLogger(__name__)


# Torch "DataSet" used for feeding data to the training routine
# Keys are... input_ids, attention_mask, target_ids, target_attention_mask
class AMRDataset(Dataset):
    def __init__(self, encodings, sents, bad_indexes):
        self.encodings   = encodings
        self.sents       = sents
        self.bad_indexes = bad_indexes  # in original file's index, not same as above

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {k:v[idx] for k, v in self.encodings.items()}


# Take a list of samples from a Dataset and collate them into a batch and returns a dict
# prepares lm_labels from target_ids, returns examples with keys as expected by the forward method
# this is necessacry because the trainer directly passes this dict as arguments to the model
# so make sure the keys match the parameter names of the forward method
# Note*1: The original code (with transformers v3.4.0) returned dict with "lm_labels".
# Support for this was removed in transformers v4.0.0 and replaced it with "labels"
class T2TDataCollator:
    def __call__(self, batch):
        input_ids = torch.stack([example['input_ids']  for example in batch])
        lm_labels = torch.stack([example['target_ids'] for example in batch])
        lm_labels[lm_labels[:, :] == 0] = -100
        attention_mask = torch.stack([example['attention_mask'] for example in batch])
        decoder_attention_mask = torch.stack([example['target_attention_mask'] for example in batch])
        return {'input_ids': input_ids, 'attention_mask': attention_mask,
                'labels': lm_labels, 'decoder_attention_mask': decoder_attention_mask }     # Note*1


# Note that for save_steps, steps means gradient updates (not batch) so if
# gradient_accumulation_steps=4 and save_steps=1000, then checkpoint is saved every 4000 batches.
class Trainer(object):
    def __init__(self, args):
        # General arguments
        self.gen_args           = args['gen_args']
        self.model_name_or_path = self.gen_args['model_name_or_path']
        self.corpus_dir         = self.gen_args['corpus_dir']
        self.train_fn           = self.gen_args['train_fn']
        self.valid_fn           = self.gen_args['valid_fn']
        self.max_in_len         = self.gen_args['max_in_len']
        self.max_out_len        = self.gen_args['max_out_len']
        # HuggingFace trainer arguments
        # See https://github.com/huggingface/transformers/blob/master/src/transformers/training_args.py
        self.training_args = TrainingArguments(**args['hf_args'])
        set_seed(self.training_args.seed)

    def train(self):
        # Create the output directory if needed
        os.makedirs(self.training_args.output_dir, exist_ok=True)
        # Load pretrained model and tokenizer
        print('Loading model and tokenizer')
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name_or_path)
        self.model     = T5ForConditionalGeneration.from_pretrained(self.model_name_or_path)
        # Clear out the "task_specific_params" and add this one
        self.model.config.task_specific_params = {'translation_amr_to_text':self.gen_args}
        # Load the datasets
        print('Building datasets')
        train_file_path = os.path.join(self.corpus_dir, self.train_fn)
        valid_file_path = os.path.join(self.corpus_dir, self.valid_fn)
        train_dataset = self.build_dataset(train_file_path)
        print('Training data is {:,} after removing {:,} long entries'.format( \
            len(train_dataset), len(train_dataset.bad_indexes)))
        valid_dataset = self.build_dataset(valid_file_path)
        print('Validation data is {:,} after removing {:,} long entries'.format( \
            len(valid_dataset), len(valid_dataset.bad_indexes)))
        # Train the model
        print('Training')
        # trainer = T5Trainer(model=self.model, args=self.training_args, train_dataset=train_dataset,
        #         eval_dataset=valid_dataset, data_collator=T2TDataCollator(), prediction_loss_only=True)
        # prediction_loss_only=True moved to training_args for compatibility with transformers v4.0.0
        trainer = T5Trainer(model=self.model, args=self.training_args, train_dataset=train_dataset,
                eval_dataset=valid_dataset, data_collator=T2TDataCollator())
        trainer.train()
        # Save the results
        print('Saving model')
        trainer.save_model(self.training_args.output_dir)
        #self.tokenizer.save_pretrained(self.training_args.output_dir)

    # Convert the AMR graphs into tokenized sentences
    def build_dataset(self, fpath):
        # Load the raw data
        entries = load_amr_graph_sent(fpath)
        # Convert to input and target sentences
        entries['input_text']  = ['%s' % graph for graph in entries['graph']]
        entries['target_text'] = ['%s' % sent  for sent  in entries['sent']]
        # Form the input encodings
        sents = entries['sent']
        input_encodings  = self.tokenizer.batch_encode_plus(entries['input_text'],
                            padding=True, truncation=True, max_length=self.max_in_len,
                            return_overflowing_tokens=True)
        target_encodings = self.tokenizer.batch_encode_plus(entries['target_text'],
                            padding=True, truncation=True, max_length=self.max_out_len,
                            return_overflowing_tokens=True)
        # Remove any graphs that are greater than max length after tokenization
        # Find the bad indexes
        bi = set()
        for i, (ie, te) in enumerate(zip(input_encodings['num_truncated_tokens'], target_encodings['num_truncated_tokens'])):
            if ie > 0 or te > 0:
                bi.add( i )
        # Remove them
        input_encodings['input_ids']  = [ie for i, ie in enumerate(input_encodings['input_ids'])  if i not in bi]
        target_encodings['input_ids'] = [te for i, te in enumerate(target_encodings['input_ids']) if i not in bi]
        input_encodings['attention_mask']  = [ie for i, ie in enumerate(input_encodings['attention_mask'])  if i not in bi]
        target_encodings['attention_mask'] = [te for i, te in enumerate(target_encodings['attention_mask']) if i not in bi]
        sents = [s  for i, s  in enumerate(sents) if i not in bi]
        # Create the encodings
        encodings = {'input_ids':             torch.LongTensor(input_encodings['input_ids']),
                     'attention_mask':        torch.LongTensor(input_encodings['attention_mask']),
                     'target_ids':            torch.LongTensor(target_encodings['input_ids']),
                     'target_attention_mask': torch.LongTensor(target_encodings['attention_mask']) }
        # Encapsulate the data and return
        return AMRDataset(encodings, sents, bi)
