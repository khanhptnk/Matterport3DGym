import os
import sys
import json
import logging
import string
import numpy as np
import random
import re
import ast
import pprint
from collections import Counter
sys.path.append('..')

from misc import util


class Dataset(object):

    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')

    def __init__(self, config, split):

        self.config = config
        self.random = random.Random(self.config.seed)
        self.split = split

        data_file = os.path.join(
            config.data_dir, config.task, config.task + '_' + split + '.json')
        self.data = self.load_data(data_file, config.vocab)

        self.item_idx = 0
        self.batch_size = config.trainer.batch_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)

    def load_data(self, data_file, vocab):

        with open(data_file) as f:
            tmp_data = json.load(f)
            data = []
            for tmp_item in tmp_data:
                for i, instr in enumerate(tmp_item['instructions']):
                    item = dict(tmp_item)
                    item['instr_id'] = str(item['path_id']) + '_' + str(i)
                    del item['path_id']
                    del item['instructions']
                    item['instruction'] = [
                        s.strip().lower() for s in self.SENTENCE_SPLIT_REGEX.split(instr.strip())
                        if len(s.strip()) > 0]

                    assert len(item['instruction']) > 0
                    data.append(item)


        logging.info('Loaded %d instances of %s split from %s' %
            (len(data), self.split, data_file))

        return data

    def iterate_batches(self, batch_size=None, data_idx=None, data_indices=None):

        if batch_size is None:
            batch_size = self.batch_size

        if data_indices is None:
            self.indices = list(range(len(self.data)))
            if self.split == 'train':
                self.random.shuffle(self.indices)
        else:
            self.indices = data_indices

        assert len(self.indices) == len(self.data)

        if data_idx is None:
            self.idx = 0
        else:
            self.idx = data_idx

        while True:
            start_idx = self.idx
            end_idx = self.idx + batch_size

            self.idx = end_idx
            if self.idx >= len(self.data):
                self.idx = 0

            batch_indices = self.indices[start_idx:end_idx]

            if len(batch_indices) < batch_size:
                batch_indices += self.random.sample(
                    self.indices, batch_size - len(batch_indices))

            batch = [self.data[i] for i in batch_indices]

            #yield self._finalize_batch(batch)
            yield batch

            if self.idx == 0 and self.split != 'train':
                break

