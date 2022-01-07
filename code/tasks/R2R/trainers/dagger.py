import logging
import os
import sys
import itertools
import json
import re
import random
import numpy as np
from collections import defaultdict
from copy import deepcopy as dc

import torch

from misc import util
import trainers.metrics as metrics


class DaggerTrainer(object):

    def __init__(self, config):

        self.config = config
        self.main_metric = self.config.trainer.main_metric
        self.max_steps = self.config.trainer.max_steps
        self.random = random.Random(self.config.seed)

    def run_episode(self, batch, env, agent, is_eval=False):

        batch_size = len(batch)

        obs, goal_descriptions = env.reset(batch, is_eval=is_eval)

        agent.init(goal_descriptions, is_eval=is_eval)

        paths = [[s.viewpoint] for s in env.states]
        reference_action_seqs = []

        steps_remaining = self.max_steps

        while not env.all_done():
            actions = agent.decide(obs)
            obs, reference_actions = env.step(actions)

            for i in range(batch_size):
                if not env.is_done[i]:
                    paths[i].append(env.states[i].viewpoint)

            reference_action_seqs.append(reference_actions)

            steps_remaining -= 1
            if steps_remaining <= 0:
                env.set_all_done()

        if not is_eval:
            loss = agent.learn(reference_action_seqs)
            return loss

        return batch, paths

    def train(self, datasets, env, agent, eval_splits):

        max_iters = self.config.trainer.max_iters
        log_every = self.config.trainer.log_every

        i_iter = 0
        total_loss = 0

        self.best_results = {}
        for split in eval_splits:
            self.best_results[split] = metrics.init_value(self.main_metric)

        for batch in datasets['train'].iterate_batches():

            i_iter += 1

            loss = self.run_episode(batch, env, agent)
            total_loss += loss

            if i_iter % log_every == 0:

                avg_loss = total_loss / log_every
                total_loss = 0

                log_str = 'Train iter %d (%d%%): ' % \
                    (i_iter, i_iter / max_iters * 100)
                log_str += 'loss = %.4f' % avg_loss

                logging.info('')
                logging.info(log_str)

                # Save last model
                agent.save('last')

                # Evaluate on held-out data
                for split in eval_splits:
                    self.evaluate(split, datasets[split], env, agent)

            if i_iter >= max_iters:
                break

    def evaluate(self, split, dataset, env, agent, pred_save_name=None):

        all_preds = {}

        for i, batch in enumerate(dataset.iterate_batches()):

            with torch.no_grad():
                batch, pred_paths = self.run_episode(batch, env, agent, is_eval=True)

            results = []
            for item, pred_path in zip(batch, pred_paths):
                results.append(metrics.eval(env.world, item['scan'], pred_path, item['path']))

            for item, pred_path, result in zip(batch, pred_paths, results):
                new_item = { 'pred_path' : pred_path,
                             'result'    : result }
                new_item.update(item)
                all_preds[new_item['instr_id']] = new_item

        avg_result = {}
        for metric_name in metrics.get_metrics():
            all_results = [item['result'][metric_name] for item in all_preds.values()]
            avg_result[metric_name] = np.average(all_results)

        log_str = 'Evaluation on %s: ' % dataset.split
        log_str += '\n  * Metrics: ' + metrics.format_results(avg_result)
        logging.info(log_str)

        # save predictions
        if pred_save_name is None:
            self.save_preds('last_%s' % split, all_preds)
        else:
            self.save_preds('%s_%s' % (split, pred_save_name), all_preds)

        if hasattr(self, 'best_results'):
            new_result = avg_result[self.main_metric]
            cur_result = self.best_results[split]
            if metrics.is_better(self.main_metric, new_result, cur_result):
                logging.info('!!! New best %s: %.3f' % (split, new_result))
                self.best_results[split] = new_result

                save_name = 'best_%s' % split
                agent.save(save_name)
                self.save_preds(save_name, all_preds)

    def save_preds(self, filename, all_preds):
        file_path = '%s/%s' % (self.config.experiment_dir, filename + '.pred')
        with open(file_path, 'w') as f:
            json.dump(all_preds, f, indent=2)
        logging.info('Saved eval info to %s' % file_path)

