import logging
import random
import torch
from   cached_property import cached_property
from   torch.utils.data import Dataset


class AMRDataset(Dataset):
    def __init__(self, tokenizer, graphs, remove_longer_than=None):
        self.tokenizer = tokenizer
        self.remove_longer_than  = remove_longer_than
        self.load_graphs(graphs)

    def load_graphs(self, graphs_in):
        self.graphs = []
        self.sentences = []
        self.linearized = []
        self.linearized_extra = []
        for g in graphs_in:
            l, e = self.tokenizer.linearize(g)
            try:
                self.tokenizer.batch_encode_sentences([g.metadata['snt']])
            except:
                logging.warning('Invalid sentence!')
                continue
            if self.remove_longer_than and len(l) > self.remove_longer_than:
                continue
            if len(l) > 1024:
                logging.warning('Sequence longer than 1024 included. BART does not support it!')
            self.sentences.append(g.metadata['snt'])
            self.graphs.append(g)
            self.linearized.append(l)
            self.linearized_extra.append(e)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sample = {}
        sample['id'] = idx
        sample['sentences'] = self.sentences[idx]
        if self.linearized is not None:
            sample['linearized_graphs_ids'] = self.linearized[idx]
            sample.update(self.linearized_extra[idx])
        return sample

    def size(self, sample):
        return len(sample['linearized_graphs_ids'])

    def collate_fn(self, samples, device):
        x = [s['sentences'] for s in samples]
        x, extra = self.tokenizer.batch_encode_sentences(x, device=device)
        if 'linearized_graphs_ids' in samples[0]:
            y = [s['linearized_graphs_ids'] for s in samples]
            y, extra_y = self.tokenizer.batch_encode_graphs_from_linearized(y, samples, device=device)
            extra.update(extra_y)
        else:
            y = None
        extra['ids'] = [s['id'] for s in samples]
        return x, y, extra


class AMRDatasetTokenBatcherAndLoader:
    def __init__(self, dataset, batch_size=800, device='cpu', shuffle=False, sort=False):
        assert not (shuffle and sort)
        self.batch_size = batch_size
        self.tokenizer = dataset.tokenizer
        self.dataset = dataset
        self.device = torch.device(device)
        self.shuffle = shuffle
        self.sort = sort

    def set_device(self, device):
        self.device = torch.device(device)

    def __iter__(self):
        it = self.sampler()
        it = ([[self.dataset[s] for s in b] for b in it])
        it = (self.dataset.collate_fn(b, device=self.device) for b in it)
        return it

    @cached_property
    def sort_ids(self):
        lengths = [len(s.split()) for s in self.dataset.sentences]
        ids, _ = zip(*sorted(enumerate(lengths), reverse=True))
        ids = list(ids)
        return ids

    def sampler(self):
        ids = list(range(len(self.dataset)))[::-1]

        if self.shuffle:
            random.shuffle(ids)
        if self.sort:
            ids = self.sort_ids.copy()

        batch_longest = 0
        batch_nexamps = 0
        batch_ntokens = 0
        batch_ids = []

        def discharge():
            nonlocal batch_longest
            nonlocal batch_nexamps
            nonlocal batch_ntokens
            ret = batch_ids.copy()
            batch_longest *= 0
            batch_nexamps *= 0
            batch_ntokens *= 0
            batch_ids[:] = []
            return ret

        while ids:
            idx = ids.pop()
            size = self.dataset.size(self.dataset[idx])
            cand_batch_ntokens = max(size, batch_longest) * (batch_nexamps + 1)
            if cand_batch_ntokens > self.batch_size and batch_ids:
                yield discharge()
            batch_longest = max(batch_longest, size)
            batch_nexamps += 1
            batch_ntokens = batch_longest * batch_nexamps
            batch_ids.append(idx)

            if len(batch_ids) == 1 and batch_ntokens > self.batch_size:
                yield discharge()

        if batch_ids:
            yield discharge()
