import warnings
warnings.simplefilter('ignore')
import os
import logging
import torch
from   torch.utils.data import Dataset
from   transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
from   transformers import TrainingArguments, DataCollatorForSeq2Seq
from   .amr_trainer import AMRTrainer
from   ...graph_processing.amr_loading import load_amr_graph_sent, load_amr_entries

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args):
        # General arguments
        self.gen_args       = args['gen_args']
        self.model_args     = args.get('model_args', {})
        self.tokenizer_args = args.get('tokenizer_args', {})
        self.training_args  = TrainingArguments(**args['hf_args'])
        set_seed(self.training_args.seed)

    def train(self):
        # Create the output directory if needed
        os.makedirs(self.training_args.output_dir, exist_ok=True)
        # Load pretrained model and tokenizer
        print('Loading model and tokenizer for', self.gen_args['model_name_or_path'])
        tokenizer_name = self.gen_args.get('tok_name_or_path', self.gen_args['model_name_or_path'])
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **self.tokenizer_args)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.gen_args['model_name_or_path'], **self.model_args)
        self.model.config.task_specific_params = {'translation_amr_to_text':self.gen_args}
        # Save the tokenizer
        if self.gen_args.get('save_tokenizer', False):
            self.tokenizer.save_pretrained(self.training_args.output_dir)
        # Load the datasets
        print('Building datasets')
        train_file_path = os.path.join(self.gen_args['corpus_dir'], self.gen_args['train_fn'])
        train_dataset   = self.build_dataset(train_file_path)
        print(f'Training data is {len(train_dataset):,} after removing {len(train_dataset.bad_indexes):,} long entries')
        # Load the evaluation dataset
        eval_fn = self.gen_args.get('eval_fn')
        if eval_fn:
            eval_fpath = os.path.join(self.gen_args['corpus_dir'], eval_fn)
            eval_samples = load_amr_entries(eval_fpath)
            print('Evaluation data is {:,} samples'.format(len(eval_samples)))
        else:
            eval_samples = None
        # Train the model
        print('Training')
        collator = DataCollatorForSeq2Seq(self.tokenizer, self.model, padding=True,
                            max_length=self.gen_args['max_train_graph_len'],
                            pad_to_multiple_of=8 if self.training_args.fp16 else None)
        trainer = AMRTrainer(model=self.model, args=self.training_args, train_dataset=train_dataset,
                             data_collator=collator, eval_tokenizer=self.tokenizer, eval_samples=eval_samples)
        trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)
        # Save the results
        if self.gen_args.get('save_at_end', False):
            print('Saving model')
            trainer.save_model(self.training_args.output_dir)

    # Convert the AMR graphs into tokenized sentences
    def build_dataset(self, fpath):
        # Load the data. Note that the input graph lines are extracted from the entry, stripped for linefeeds
        # and leading/trailing spaces then rejoined by a space. This is basically the same as in inference except
        # for removal of multiple spaces inside the graph, which shouldn't be there anyway.
        entries = load_amr_graph_sent(fpath)
        # Convert to input and target sentences
        entries['input_text']  = ['%s' % graph for graph in entries['graph']]
        entries['target_text'] = ['%s' % sent  for sent  in entries['sent']]
         # tokenize with verbose=False to eliminate messages about too long for model
        inp_enc = self.tokenizer(entries['input_text'],  verbose=False)
        tgt_enc = self.tokenizer(entries['target_text'], verbose=False)
        # Remove any graphs that are greater than max length after tokenization
        bi  = set(i for i, ids in enumerate(inp_enc['input_ids']) if len(ids) > self.gen_args['max_train_graph_len'])
        bi |= set(i for i, ids in enumerate(tgt_enc['input_ids']) if len(ids) > self.gen_args['max_train_sent_len'])
        # Remove them
        encodings = {}
        encodings['input_ids']      = [e for i, e in enumerate(inp_enc['input_ids'])      if i not in bi]
        encodings['attention_mask'] = [e for i, e in enumerate(inp_enc['attention_mask']) if i not in bi]
        encodings['labels']         = [e for i, e in enumerate(tgt_enc['input_ids'])      if i not in bi]
        sents = [s  for i, s  in enumerate(entries['sent']) if i not in bi]
        return AMRDataset(encodings, sents, bi)


# Torch "DataSet" used for feeding data to the training routine
class AMRDataset(Dataset):
    def __init__(self, encodings, sents, bad_indexes):
        self.encodings   = encodings
        self.sents       = sents
        self.bad_indexes = bad_indexes  # in original file's index, not same as above

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {k:v[idx] for k, v in self.encodings.items()}
