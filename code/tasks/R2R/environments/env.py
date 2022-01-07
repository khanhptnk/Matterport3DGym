import sys
import os
import random
import math

import worlds


class Environment(object):

    STOP_ACTION = 0

    def __init__(self, config):

        self.config = config
        self.world = worlds.load(config)
        self.random = random.Random(config.seed)

    def get_path(self, s, g):
        return self.world.get_shortest_path(s.scan, s.viewpoint, g)

    def all_done(self):
        return all(self.is_done)

    def set_all_done(self):
        for i in range(self.batch_size):
            self.is_done[i] = True

    def make_obs(self):
        obs = []
        for i, (s, g) in enumerate(zip(self.states, self.goals)):
            obs.append({
                'view_features': s.curr_view_features,
                'action_embeddings': s.action_embeddings,
                'heading': s.heading,
                'elevation': s.elevation,
            })

        return obs

    def reset(self, batch, is_eval=False):

        self.batch_size = len(batch)

        self.is_eval = is_eval

        self.is_done = [False] * self.batch_size

        poses = []
        for item in batch:
            poses.append((item['scan'], item['path'][0], item['heading'], 0))
        self.states = self.world.init(poses)

        self.goals = [item['path'][-1] for item in batch]
        goal_descriptions = [item['instruction'] for item in batch]

        obs = self.make_obs()

        return obs, goal_descriptions

    def get_reference_actions(self):

        ref_actions = []

        for s, g in zip(self.states, self.goals):

            if s.viewpoint == g:
                ref_actions.append(self.STOP_ACTION)
            else:
                next_viewpoint = self.get_path(s, g)[1]
                for i, loc in enumerate(s.adj_loc_list):
                    if loc['nextViewpointId'] == next_viewpoint:
                        ref_actions.append(i)
                        break

        return ref_actions

    def step(self, actions):

        ref_actions = self.get_reference_actions()
        for i in range(self.batch_size):
            if self.is_done[i]:
                ref_actions[i] = -1
                actions[i] = self.STOP_ACTION

        self.states = self.states.step(actions)
        obs = self.make_obs()

        for i in range(self.batch_size):
            self.is_done[i] |= (actions[i] == self.STOP_ACTION)

        return obs, ref_actions


