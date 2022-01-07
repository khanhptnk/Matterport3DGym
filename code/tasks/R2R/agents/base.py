import os
import sys
import json
import logging
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import models
import worlds


class Agent(object):

    def _to_tensor(self, x, from_numpy=False):
        if from_numpy:
             return torch.from_numpy(x).to(self.device)
        return torch.tensor(x).to(self.device)

    def load_model(self, config):
        raise NotImplementedError

    def init(self, goal_descriptions, is_eval=False):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def decide(self, obs):
        raise NotImplementedError

    def save(self, name, trajectories=None):
        file_path = os.path.join(self.config.experiment_dir, name + '.ckpt')
        ckpt = { 'model_state_dict': self.model.state_dict(),
                 'optim_state_dict': self.optim.state_dict() }
        torch.save(ckpt, file_path)
        logging.info('Saved %s model to %s' % (name, file_path))

    def load(self, file_path):
        ckpt = torch.load(file_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optim.load_state_dict(ckpt['optim_state_dict'])
        logging.info('Loaded model from %s' % file_path)





