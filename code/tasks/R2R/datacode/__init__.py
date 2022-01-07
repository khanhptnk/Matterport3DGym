import os
import json
import logging

from misc import util

from .dataset import Dataset


def load_vocab(config):

    data_dir = config.data_dir

    vocab_file = os.path.join(data_dir, config.task, 'vocab.json' or config.vocab_file)
    if os.path.exists(vocab_file):
        with open(vocab_file) as f:
            words = json.load(f)
    else:
        raise Exception('%s not found' % vocab_file)

    if '<CLS>' not in words:
        words.append('<CLS>')

    assert '<PAD>' in words
    assert '<EOS>' in words

    vocab = util.Vocab()
    for w in words:
        vocab.index(w)

    logging.info('Loaded word vocab of size %d' % len(vocab))

    return vocab

def load(config):

    config.vocab = load_vocab(config)

    datasets = {}
    splits = ['train', 'val_seen', 'val_unseen', 'test']
    for split in splits:
        datasets[split] = Dataset(config, split)

    return datasets

