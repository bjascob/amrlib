import warnings
warnings.simplefilter('ignore')
import os
import time
import random
import traceback
from   datetime import datetime
import torch
from   torch.optim import AdamW
from   .modules.parser import Parser
from   .data_loader import DataLoader
from   .vocabs import get_vocabs
from   .utils import move_to_device
from   .bert_utils import BertEncoderTokenizer, BertEncoder
from   .inference import Inference


# LR scheduler = lr_scale(1.0) * 1/sqrt(512) * min(1/sqrt(batchnum), (batchnum/warmup)*1/sqrt(warmup))
# This gives a curve that rises for "warmup_steps" (step ==> count of the total number of batches)
# and then falls slowly after that
def update_lr(optimizer, lr_scale, embed_size, steps, warmup_steps):
    lr = lr_scale * embed_size**-0.5 * min(steps**-0.5, steps*(warmup_steps**-1.5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def run_training(args, ls):
    ls.print('Training started: ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    # Misc setup
    os.makedirs(args.model_dir, exist_ok=True)
    assert len(args.cnn_filters)%2 == 0
    args.cnn_filters = list(zip(args.cnn_filters[:-1:2], args.cnn_filters[1::2]))
    # Load the vocabs
    vocabs = get_vocabs(os.path.join(args.model_dir, args.vocab_dir))
    bert_tokenizer = None
    if args.with_bert:
        bert_tokenizer = BertEncoderTokenizer.from_pretrained(args.bert_path, do_lower_case=False)
        vocabs['bert_tokenizer'] = bert_tokenizer
    for name in vocabs:
        if name == 'bert_tokenizer':
            continue
        ls.print('Vocab %-20s  size %5d  coverage %.3f' % (name, vocabs[name].size, vocabs[name].coverage))
    # Setup BERT encoder
    bert_encoder = None
    if args.with_bert:
        bert_encoder = BertEncoder.from_pretrained(args.bert_path)
        for p in bert_encoder.parameters():
            p.requires_grad = False
    # Device and random setup
    torch.manual_seed(19940117)
    torch.cuda.manual_seed_all(19940117)
    random.seed(19940117)
    device = torch.device(args.device)
    # Create the model
    ls.print('Setting up the model')
    model = Parser(vocabs,
            args.word_char_dim, args.word_dim, args.pos_dim, args.ner_dim,
            args.concept_char_dim, args.concept_dim,
            args.cnn_filters, args.char2word_dim, args.char2concept_dim,
            args.embed_dim, args.ff_embed_dim, args.num_heads, args.dropout,
            args.snt_layers, args.graph_layers, args.inference_layers, args.rel_dim,
            device, args.pretrained_file, bert_encoder,)
    model = model.to(device)
    # Optimizer and weight decay params
    weight_decay_params = []
    no_weight_decay_params = []
    for name, param in model.named_parameters():
        if name.endswith('bias') or 'layer_norm' in name:
            no_weight_decay_params.append(param)
        else:
            weight_decay_params.append(param)
    grouped_params = [{'params':weight_decay_params, 'weight_decay':1e-4},
                        {'params':no_weight_decay_params, 'weight_decay':0.}]
    optimizer = AdamW(grouped_params, 1., betas=(0.9, 0.999), eps=1e-6)
    # Re-load an existing model if requested
    used_batches = 0
    batches_acm = 0
    if args.resume_ckpt:
        ls.print('Resuming from checkpoint', args.resume_ckpt)
        ckpt = torch.load(args.resume_ckpt)
        model.load_state_dict(ckpt['model'])
        if ckpt.get('optimizer', {}):
            optimizer.load_state_dict(ckpt['optimizer'])
        else:
            ls.print('No optimizer state saved in checkpoint, using default initial optimizer')
        batches_acm = ckpt['batches_acm']
        start_epoch = ckpt['epoch'] + 1
        del ckpt
    else:
        start_epoch = 1     # don't start at 0
    # Load data
    ls.print('Loading training data')
    train_data = DataLoader(vocabs, args.train_data, args.train_batch_size, for_train=True)
    train_data.set_unk_rate(args.unk_rate)
    # Train
    ls.print('Training')
    epoch, loss_avg, concept_loss_avg, arc_loss_avg, rel_loss_avg = 0, 0, 0, 0, 0
    for epoch in range(start_epoch, args.epochs+1):
        st = time.time()
        for batch in train_data:
            model.train()
            batch = move_to_device(batch, model.device)
            concept_loss, arc_loss, rel_loss, graph_arc_loss = model(batch)
            loss = (concept_loss + arc_loss + rel_loss) / args.batches_per_update
            loss_value = loss.item()
            concept_loss_value = concept_loss.item()
            arc_loss_value = arc_loss.item()
            rel_loss_value = rel_loss.item()
            loss_avg = loss_avg * args.batches_per_update * 0.8 + 0.2 * loss_value
            concept_loss_avg = concept_loss_avg * 0.8 + 0.2 * concept_loss_value
            arc_loss_avg = arc_loss_avg * 0.8 + 0.2 * arc_loss_value
            rel_loss_avg = rel_loss_avg * 0.8 + 0.2 * rel_loss_value
            loss.backward()
            used_batches += 1
            if not (used_batches % args.batches_per_update == -1 % args.batches_per_update):
                continue
            batches_acm += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            lr = update_lr(optimizer, args.lr_scale, args.embed_dim, batches_acm, args.warmup_steps)
            optimizer.step()
            optimizer.zero_grad()
        # Summary at the end of the epoch
        dur = time.time() - st
        ls.print('Epoch %4d, Batch %5d, LR %.6f, conc_loss %.3f, arc_loss %.3f, rel_loss %.3f, duration %.1f seconds' %
                    (epoch, batches_acm, lr, concept_loss_avg, arc_loss_avg, rel_loss_avg, dur))
        # Evaluate and save the data every so often
        if (epoch>args.skip_evals or args.resume_ckpt is not None) and epoch % args.eval_every == 0:
            model.eval()
            ls.print('Evaluating and saving the model')
            fname = '%s/epoch%d.pt'%(args.model_dir, epoch)
            optim = optimizer.state_dict() if args.save_optimizer else {}
            torch.save({'args':vars(args), 'model':model.state_dict(), 'batches_acm': batches_acm,
                        'optimizer': optim, 'epoch':epoch}, fname)
            try:
                out_fn = 'epoch%d.pt.dev_generated' % (epoch)
                inference = Inference.build_from_model(model, vocabs)
                f_score, ctr = inference.reparse_annotated_file('.', args.dev_data, args.model_dir, out_fn,
                        print_summary=False)
                ls.print('Smatch F: %.3f.  Wrote %d AMR graphs to %s' % \
                        (f_score, ctr, os.path.join(args.model_dir, out_fn)))
            except:
                ls.print('Exception during generation')
                traceback.print_exc()
            model.train()
    # End time-stamp
    ls.print('Training finished: ' + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
