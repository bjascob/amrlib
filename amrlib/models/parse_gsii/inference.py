import warnings
warnings.simplefilter('ignore')
import os
import io
import logging
import penman
import torch
from   tqdm import tqdm
from   .modules.parser import Parser
from   .data_loader import DataLoader
from   .vocabs import get_vocabs
from   .graph_builder import GraphBuilder
from   .utils import move_to_device
from   .bert_utils import BertEncoderTokenizer, BertEncoder
from   ..inference_bases import STOGInferenceBase
from   ...graph_processing.amr_loading import load_amr_entries, split_amr_meta
from   ...graph_processing.annotator import annotate_graph
from   ...evaluate.smatch_enhanced import compute_smatch
from   ...utils.config import Config


logger = logging.getLogger(__name__)


class Inference(STOGInferenceBase):
    def __init__(self, model_dir, model_fn, **kwargs):
        self.model_dir       = model_dir
        self.model_fn        = model_fn
        default_device       = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        device               = kwargs.get('device', default_device)
        self.device          = torch.device(device)
        self.batch_size      = kwargs.get('batch_size',   6000)   # note that batch-size is in tokens
        self.beam_size       = kwargs.get('beam_size',       8)
        self.alpha           = kwargs.get('alpha',         0.6)
        self.max_time_step   = kwargs.get('max_time_step', 100)
        if model_fn:
            self._load_model()  # sets self.model, graph_builder, vocabs

    # When training use the existing model and vocabs
    @classmethod
    def build_from_model(cls, model, vocabs, **kwargs):
        self = cls(None, None, **kwargs)
        self.model         = model
        self.device        = model.device
        self.vocabs        = vocabs
        self.graph_builder = GraphBuilder(vocabs['rel'])
        return self

    # parse a list of sentences (strings)
    def parse_sents(self, sents, add_metadata=True):
        assert isinstance(sents, list)
        # Annotate the entry then compile it in a StringIO file-type object
        # For Simplicity, convert the sentences into an in-memory AMR text file
        # This could be simplified but the DataLoader is setup to load from an AMR file
        # and this method will create an in-memory file-type object in the AMR format.
        sio_f = io.StringIO()
        for i, sent in enumerate(sents):
            sent = sent.replace('\n', ' ')  # errant line-feeds will confuse the parser
            entry  = '# ::snt %s\n' % sent
            entry += '(d / dummy)\n'        # not-used but required for proper AMR file format
            pen_graph = annotate_graph(entry)
            amr_string = penman.encode(pen_graph)
            sio_f.write(amr_string + '\n')
            if i != len(sents)-1:
                sio_f.write('\n')
        sio_f.seek(0)
        return self.parse_file_handle(sio_f, add_metadata)

    # parse a list of spacy spans (ie.. span has list of tokens)
    # Duplicate of above code but SpaCy parsing is already done so don't do it again
    def parse_spans(self, spans, add_metadata=True):
        sio_f = io.StringIO()
        for i, span in enumerate(spans):
            sent   = span.text
            tokens = list(span)
            entry  = '# ::snt %s\n' % sent
            entry += '(d / dummy)\n'        # not-used but required for proper AMR file format
            pen_graph = annotate_graph(entry, tokens)
            amr_string = penman.encode(pen_graph)
            sio_f.write(amr_string + '\n')
            if i != len(spans)-1:
                sio_f.write('\n')
        sio_f.seek(0)
        return self.parse_file_handle(sio_f, add_metadata)

    # Parse an open file handle for a properly annotated AMR graph
    # annotations must include, tokens, lemmas, pos_tags, ner_tags and ner_iob
    # This function probably should not be called directly.  Use parse_sents or parse_spans above
    def parse_file_handle(self, sio_f, add_metadata=True):
        # Create the DataLoader and parser the data
        data_loader = DataLoader(self.vocabs, sio_f, self.batch_size, for_train=False)
        output_entries = []
        for batch in data_loader:
            batch = move_to_device(batch, self.model.device)
            res = self._parse_batch(batch)
            for concept, relation in zip(res['concept'], res['relation']):
                graph_lines = self.graph_builder.build(concept, relation)
                output_entries.append( graph_lines )
        # Add the metadata from the annotations and 'snt' if requested
        if add_metadata:
            sio_f.seek(0)
            entries = sio_f.read().split('\n\n')
            assert len(entries) == len(output_entries)
            meta_lines = [split_amr_meta(e)[0] for e in entries]    # get the metadata as a list of lines
            meta_lines = ['\n'.join(ml) for ml in meta_lines]       # join lines into a string for each entry
            output_entries = [a + '\n' + b for a, b in zip(meta_lines, output_entries)]
        return output_entries

    # Parse the AMR input file and then add the meta-data from here to the final output.
    # Compute smatch scores between the input_file and the generated graphs.
    def reparse_annotated_file(self, indir, infn, outdir, outfn, print_summary=True):
        # Load the test data and the model
        test_data_fn = os.path.join(indir, infn)
        output_fn    = os.path.join(outdir, outfn)
        print('Loading test data from ', test_data_fn)
        test_data = DataLoader(self.vocabs, test_data_fn, self.batch_size, for_train=False)
        # Load the reference amr file that contains all the metadata
        entries = load_amr_entries(test_data_fn)    # Note - already loaded above, but simplest for now.
        # Loop through the test_data batches and generate, then write out generated data
        ctr, gold_entries, test_entries = 0, [], []
        pbar = tqdm(total=len(entries))
        with open(output_fn, 'w') as fo:
            for batch in test_data:
                batch = move_to_device(batch, self.model.device)
                res = self._parse_batch(batch)
                for concept, relation, score in zip(res['concept'], res['relation'], res['score']):
                    # Write the original metadata and keep the graphs for scoring at the bottom
                    meta_lines, graph_lines = split_amr_meta(entries[ctr])
                    gold_entries.append(' '.join(graph_lines))
                    for line in meta_lines:
                        fo.write(line + '\n')
                    # Write some new metadata   - for test
                    graph_lines = self.graph_builder.build(concept, relation)
                    test_entries.append(' '.join(graph_lines.splitlines()))
                    fo.write(graph_lines + '\n\n')
                    ctr += 1
                    pbar.update(1)
        pbar.close()
        # Compute smatch score
        try:
            precision, recall, f_score = compute_smatch(test_entries, gold_entries)
        except:
            logger.error('compute_smatch failed')
            f_score = 0
        if print_summary:
            print('Smatch F: %.3f.  Wrote %d AMR graphs to %s' % (f_score, ctr, output_fn))
        return f_score, ctr

    # Process a batch
    def _parse_batch(self, batch):
        res = dict()
        concept_batch = []
        relation_batch = []
        beams = self.model.work(batch, self.beam_size, self.max_time_step)
        score_batch = []
        for beam in beams:
            best_hyp = beam.get_k_best(1, self.alpha)[0]
            predicted_concept = [token for token in best_hyp.seq[1:-1]]
            predicted_rel = []
            for i in range(len(predicted_concept)):
                if i == 0:
                    continue
                arc = best_hyp.state_dict['arc_ll%d'%i].squeeze_().exp_()[1:]   # head_len
                rel = best_hyp.state_dict['rel_ll%d'%i].squeeze_().exp_()[1:,:] # head_len x vocab
                for head_id, (arc_prob, rel_prob) in enumerate(zip(arc.tolist(), rel.tolist())):
                    predicted_rel.append((i, head_id, arc_prob, rel_prob))
            concept_batch.append(predicted_concept)
            score_batch.append(best_hyp.score)
            relation_batch.append(predicted_rel)
        res['concept'] = concept_batch
        res['score'] = score_batch
        res['relation'] = relation_batch
        return res

    # Load the model, post-proc and vocabs.
    def _load_model(self):
        model_fpath = os.path.join(self.model_dir, self.model_fn)
        print('Loading model', model_fpath)
        model_dict = torch.load(model_fpath, map_location='cpu')    # always load initially to RAM
        model_args = Config(model_dict['args'])
        vocabs = get_vocabs(os.path.join(self.model_dir, model_args.vocab_dir))
        # Create the post-processor
        graph_builder = GraphBuilder(vocabs['rel'])
        # Load bert of specified
        bert_encoder = None
        if model_args.with_bert:
            bert_tokenizer = BertEncoderTokenizer.from_pretrained(model_args.bert_path, do_lower_case=False)
            bert_encoder = BertEncoder.from_pretrained(model_args.bert_path)
            vocabs['bert_tokenizer'] = bert_tokenizer
        # Setup the model
        model = Parser(vocabs,
                model_args.word_char_dim, model_args.word_dim, model_args.pos_dim, model_args.ner_dim,
                model_args.concept_char_dim, model_args.concept_dim,
                model_args.cnn_filters, model_args.char2word_dim, model_args.char2concept_dim,
                model_args.embed_dim, model_args.ff_embed_dim, model_args.num_heads, model_args.dropout,
                model_args.snt_layers, model_args.graph_layers, model_args.inference_layers, model_args.rel_dim,
                device=self.device, bert_encoder=bert_encoder)
        # Load the trained model's values
        state_dict = model_dict['model']
        # The following was in the original code but I'm not sure why and it doesn't seem to make any difference
        # if it's taken out.  It only seems to keep from loading the same pretrained values twice.
        # Load the checkpoint and replace the saved model's bert_encoder values with
        # the pretrained values from the empty model above
        for k, v in model.state_dict().items():
            if k.startswith('bert_encoder'):
                state_dict[k] = v
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        # Set instance variables
        self.vocabs        = vocabs
        self.graph_builder = graph_builder
        self.model         = model
