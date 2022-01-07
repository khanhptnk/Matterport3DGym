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
import flags

from misc import util
from agents.base import Agent


class NavAgentPREVALENT(Agent):

    def __init__(self, config):

        self.config = config
        self.device = config.device
        self.random = random.Random(self.config.seed)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab = config.vocab

        self.model_config = config.nav_agent.model
        self.model_config.pad_idx = self.vocab['<PAD>']

        self.load_model(config)

    def load_model(self, config):

        self.model = models.load(config).to(self.device)

        logging.info('nav model: ' + str(self.model))

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.model_config.lr)

        if config.nav_agent.model.load_from is not None:
            self.load(config.nav_agent.model.load_from)

    def init(self, goal_descriptions, is_eval=False):

        if is_eval:
            self.model.eval()
        else:
            self.model.train()

        self.is_eval = is_eval
        self.batch_size = len(goal_descriptions)

        self.reset()

        self.encode_goals(goal_descriptions)

    def reset(self):
        self.action_logit_seqs = []
        self.prev_actions = None

    def _length2mask(self, length, size=None):
        batch_size = len(length)
        size = int(max(length)) if size is None else size
        mask = (torch.arange(size, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
                > (torch.LongTensor(length) - 1).unsqueeze(1)).to(self.device)
        return mask

    def _sort_batch(self, instructions):

        pad_idx = self.model_config.pad_idx

        for i, instr in enumerate(instructions):
            instr = instr[:self.model_config.max_instruction_length]
            instr = ['<CLS>'] + instr + ['<EOS>']
            instructions[i] = [self.vocab[w] for w in instr]
        max_len = max([len(instr) for instr in instructions])

        for instr in instructions:
            while len(instr) < max_len:
                instr.append(pad_idx)

        seq_tensor = np.array(instructions)

        seq_lengths = np.argmax(seq_tensor == pad_idx, axis=1)
        seq_lengths[seq_lengths == 0] = seq_tensor.shape[1]

        seq_tensor = self._to_tensor(seq_tensor, from_numpy=True).long()
        seq_lengths = self._to_tensor(seq_lengths, from_numpy=True)

        # Sort sequences by lengths
        seq_lengths, perm_idx = seq_lengths.sort(0, True)  # True -> descending
        sorted_tensor = seq_tensor[perm_idx]
        mask = (sorted_tensor != pad_idx).long()

        token_type_ids = torch.zeros_like(mask)

        return sorted_tensor, mask, token_type_ids, seq_lengths.tolist(), perm_idx.tolist()

    def encode_goals(self, goal_descriptions):

        sentence, self.language_attention_mask, self.token_type_ids, \
            seq_lengths, self.perm_idx = self._sort_batch(goal_descriptions)

        language_inputs = {'mode'          : 'language',
                           'sentence'      : sentence,
                           'attention_mask': self.language_attention_mask,
                           'lang_mask'     : self.language_attention_mask,
                           'token_type_ids': self.token_type_ids}

        self.h_t, self.language_features = self.model(**language_inputs)

    def _candidate_variable(self, obs):
        candidate_leng = [len(ob['action_embeddings']) for ob in obs]  # +1 is for the end
        candidate_feat = np.zeros((len(obs), max(candidate_leng),
            self.model_config.image_feat_size + self.model_config.angle_feat_size), dtype=np.float32)
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            num_a = ob['action_embeddings'].shape[0]
            candidate_feat[i, :num_a, :] = ob['action_embeddings']

        return self._to_tensor(candidate_feat, from_numpy=True), candidate_leng

    def _get_input_feat(self, obs):
        input_a_t = np.zeros((len(obs), self.model_config.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            input_a_t[i] = util.build_angle_features(
                ob['heading'], ob['elevation'], self.model_config.angle_feat_size)
        input_a_t = self._to_tensor(input_a_t, from_numpy=True)
        # f_t = self._feature_variable(obs)      # Pano image features from obs
        candidate_feat, candidate_leng = self._candidate_variable(obs)

        return input_a_t, candidate_feat, candidate_leng

    def decide(self, obs):

        obs = np.array(obs)
        perm_obs = obs[self.perm_idx]

        input_a_t, candidate_feat, candidate_leng = self._get_input_feat(perm_obs)

        # the first [CLS] token, initialized by the language BERT, serves
        # as the agent's state passing through time steps
        self.language_features = torch.cat((self.h_t.unsqueeze(1), self.language_features[:,1:,:]), dim=1)

        visual_temp_mask = (self._length2mask(candidate_leng) == 0).long()
        visual_attention_mask = torch.cat((self.language_attention_mask, visual_temp_mask), dim=-1)

        self.model.vln_bert.config.directions = max(candidate_leng)

        ''' Visual BERT '''
        visual_inputs = {'mode':              'visual',
                        'sentence':           self.language_features,
                        'attention_mask':     visual_attention_mask,
                        'lang_mask':          self.language_attention_mask,
                        'vis_mask':           visual_temp_mask,
                        'token_type_ids':     self.token_type_ids,
                        'action_feats':       input_a_t,
                        # 'pano_feats':         f_t,
                        'cand_feats':         candidate_feat}

        self.h_t, logit = self.model(**visual_inputs)
        candidate_mask = self._length2mask(candidate_leng)
        logit.masked_fill_(candidate_mask, -float('inf'))

        # permute back
        action_logits = torch.zeros_like(logit)
        for i, idx in enumerate(self.perm_idx):
            action_logits[idx] = logit[i]

        self.action_logit_seqs.append(action_logits)

        if self.is_eval:
            pred_actions = action_logits.max(dim=1)[1].tolist()
        else:
            pred_actions = D.Categorical(logits=action_logits).sample().tolist()

        return pred_actions

    def compute_loss(self, ref_action_seqs):

        assert len(ref_action_seqs) == len(self.action_logit_seqs)

        loss = 0
        for ref_action, logit in zip(ref_action_seqs, self.action_logit_seqs):
            ref_action = self._to_tensor(ref_action).long()
            loss += self.loss_fn(logit, ref_action)

        return loss

    def learn(self, ref_action_seqs):

        loss = self.compute_loss(ref_action_seqs)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item() / len(ref_action_seqs)


