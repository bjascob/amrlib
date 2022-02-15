import warnings
warnings.simplefilter('ignore')
import os
import logging
import torch
from   torch.utils.data import Dataset
from   transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
from   transformers import TrainingArguments, TrainerCallback, DataCollatorForSeq2Seq
from   .amr_trainer import AMRTrainer
from   .penman_serializer import load_and_serialize

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args):
        # General arguments
        self.gen_args      = args['gen_args']
        self.model_args    = args.get('model_args', {})
        self.training_args = TrainingArguments(**args['hf_args'])
        set_seed(self.training_args.seed)

    def train(self):
        # Create the output directory if needed
        os.makedirs(self.training_args.output_dir, exist_ok=True)
        # Load pretrained model and tokenizer
        print('Loading model and tokenizer for', self.gen_args['model_name_or_path'])
        tokenizer_name = self.gen_args.get('tok_name_or_path', self.gen_args['model_name_or_path'])
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model     = AutoModelForSeq2SeqLM.from_pretrained(self.gen_args['model_name_or_path'], **self.model_args)
        self.model.config.task_specific_params = {'parse_amr':self.gen_args}
        assert self.model.config.no_repeat_ngram_size == 0  # required for good results
        # `use_cache=True` is incompatible with gradient checkpointing
        if self.training_args.gradient_checkpointing:
            self.model.config.use_cache=False
        # Save the tokenizer
        if self.gen_args.get('save_tokenizer', False):
            self.tokenizer.save_pretrained(self.training_args.output_dir)
        # Load the datasets
        print('Building datasets')
        train_file_path = os.path.join(self.gen_args['corpus_dir'], self.gen_args['train_fn'])
        train_dataset   = self.build_dataset(train_file_path)
        print('Training data is {:,} after removing {:,} long entries'.format( \
            len(train_dataset), len(train_dataset.bad_indexes)))
        # Load the evaluation dataset
        eval_fn = self.gen_args.get('eval_fn')
        if eval_fn:
            eval_file_path = os.path.join(self.gen_args['corpus_dir'], eval_fn)
            eval_samples   = load_and_serialize(eval_file_path)
            print('Evaluation data is {:,} samples'.format(len(eval_samples['graphs'])))
        else:
            eval_samples = None
        # Train the model
        print('Training')
        collator = DataCollatorForSeq2Seq(self.tokenizer, self.model, padding=True,
                        max_length=self.gen_args['max_train_graph_len'],
                        pad_to_multiple_of=8 if self.training_args.fp16 else None)
        trainer = AMRTrainer(model=self.model, args=self.training_args, train_dataset=train_dataset,
                             data_collator=collator, eval_tokenizer=self.tokenizer, eval_samples=eval_samples,
                             callbacks=[CudaMemoryCB()])
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
        # tokenize with verbose=False to eliminate messages about too long for model
        print('Tokenizing')
        inp_enc = self.tokenizer(entries['sents'],   verbose=False)
        tgt_enc = self.tokenizer(entries['serials'], verbose=False, add_special_tokens=False)
        # When setting add_special_tokens=False, the tokenizer will not add bos or eos tokens to the output.
        # T5 has no bos token anyway but bart does (id=0) and we don't want it in the labels.
        # Because the decoder_input_ids are the labels shifted right with the last token truncated,
        # we want an eos token at the end so that a word token is not truncated.
        tgt_enc['input_ids'] = [seq + [self.tokenizer.eos_token_id] for seq in tgt_enc['input_ids']]
        # A bos token might be needed for some future seq2seq model, though not for bart or t5.
        if self.gen_args.get('add_bos_token_to_labels', False):
            tgt_enc['input_ids'] = [[self.tokenizer.bos_token_id] + seq for seq in tgt_enc['input_ids']]

        # Identify any truncated data and strip it
        bi  = set(i for i, ids in enumerate(inp_enc['input_ids']) if len(ids) > self.gen_args['max_train_sent_len'])
        bi |= set(i for i, ids in enumerate(tgt_enc['input_ids']) if len(ids) > self.gen_args['max_train_graph_len'])
        # Compile the output encodings, stripped of bad indexes
        # These will be passed directly to the model so make sure all the keys are correct, with no extras
        encodings = {}
        encodings['input_ids']      = [e for i, e in enumerate(inp_enc['input_ids'])      if i not in bi]
        encodings['attention_mask'] = [e for i, e in enumerate(inp_enc['attention_mask']) if i not in bi]
        encodings['labels']         = [e for i, e in enumerate(tgt_enc['input_ids'])      if i not in bi]
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


# https://pytorch.org/docs/stable/generated/torch.cuda.memory_stats.html
class CudaMemoryCB(TrainerCallback):
    def on_log(self, args, state, control, logs={}, **kwargs):
        # Here just to clean-up print alignment to make things easier to read
        if 'learning_rate' in logs and 'loss' in logs and 'epoch' in logs:
            logs['learning_rate'] = '%.2e' % logs['learning_rate']
            logs['epoch']         = '%.2f' % logs['epoch']
            logs['loss']          = '%.4f' % logs['loss']
        # Log the cuda memory stats
        stats = self.cuda_memory_stats()
        logs['cuda'] = stats

    def on_epoch_begin(self, args, state, control, logs={}, **kwargs):
        device = torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(device)

    def cuda_memory_stats(self):
        device  = torch.cuda.current_device()
        stats   = torch.cuda.memory_stats(device)
        string  = 'alloc=%s/%s' % (self.format_bytes(stats['allocated_bytes.all.current']),
                                   self.format_bytes(stats['allocated_bytes.all.peak']))
        string += ', '
        string += 'res=%s/%s'   % (self.format_bytes(stats['reserved_bytes.all.current']),
                                   self.format_bytes(stats['reserved_bytes.all.peak']))
        return string

    @staticmethod
    def format_bytes(size):
        power_labels = {0 : '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
        n = 0
        while size > 1024:
            size /= 1024
            n += 1
        return '%.1f%s' % (size, power_labels.get(n,'X'))
