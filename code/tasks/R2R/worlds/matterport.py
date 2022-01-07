import os
import sys
import argparse
import csv
import math
import json
import random
import logging
from collections import defaultdict

import networkx as nx
import numpy as np
import torch

from misc import util
import MatterSim


csv.field_size_limit(sys.maxsize)

angle_inc = math.pi / 6.
NUM_VIEWS = 36

IMAGE_W = 640
IMAGE_H = 480
VFOV = 60


class ImageFeatures(object):

    def __init__(self, image_feature_file, image_feat_size, angle_feat_size):

        logging.info('Loading image features from %s' % image_feature_file)
        tsv_fieldnames = ['scanId', 'viewpointId', 'image_w','image_h', 'vfov', 'features']

        default_features = np.zeros((NUM_VIEWS, image_feat_size), dtype=np.float32)
        self.features = defaultdict(lambda: default_features)

        with open(image_feature_file, "rt") as tsv_in_file:
            reader = csv.DictReader(
                tsv_in_file, delimiter='\t', fieldnames=tsv_fieldnames)
            for item in reader:

                assert int(item['image_h']) == IMAGE_H
                assert int(item['image_w']) == IMAGE_W
                assert int(item['vfov']) == VFOV

                long_id = item['scanId'] + '_' + item['viewpointId']
                features = np.frombuffer(util.decode_base64(item['features']),
                    dtype=np.float32).reshape((NUM_VIEWS, image_feat_size))
                self.features[long_id] = features

    def get_features(self, scan, viewpoint):
        long_id = scan + '_' + viewpoint
        return self.features[long_id]


class MatterportWorldMeta(object):

    def __init__(self, config):

        self.config = config
        self.angle_feat_size = config.nav_agent.model.angle_feat_size
        self.image_feat_size = config.nav_agent.model.image_feat_size

        self.load_graphs()
        self.load_cached_adjacent_lists()
        self.load_image_features()
        self.make_angle_features()

    def load_graphs(self):
        scan_list_file = os.path.join(
            self.config.data_dir, 'connectivity/scans.txt')
        scans = set(open(scan_list_file).read().strip().split('\n'))
        self.graphs = {}
        self.paths = {}
        self.distances = {}
        for scan in scans:
            graph = util.load_nav_graphs(scan, self.config.data_dir)
            self.graphs[scan] = graph
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(graph))
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(graph))

    def load_cached_adjacent_lists(self):
        cached_action_space_path = os.path.join(
            self.config.data_dir, 'panoramic_action_space.json')
        with open(os.path.join(cached_action_space_path)) as f:
            self.cached_adj_loc_lists = json.load(f)

    def load_image_features(self):
        image_feature_path = os.path.join(self.config.data_dir, 'img_features',
            self.config.world.image_feature_file)
        self.featurizer = ImageFeatures(image_feature_path, self.image_feat_size, self.angle_feat_size)

    def make_angle_features(self):
        self.angle_features = []
        for i in range(NUM_VIEWS):
            embeddings = np.zeros((NUM_VIEWS, self.angle_feat_size), dtype=np.float32)
            base_heading = (i % 12) * angle_inc
            for j in range(NUM_VIEWS):
                heading = (j % 12) * angle_inc
                heading -= base_heading
                elevation = ((j // 12) - 1) * angle_inc
                embeddings[j, :] = util.build_angle_features(heading, elevation, self.angle_feat_size)
            self.angle_features.append(embeddings)


class MatterportWorld(object):

    STOP_ACTION = 0

    def __init__(self, meta):

        self.config = meta.config

        self.featurizer = meta.featurizer
        self.graphs = meta.graphs
        self.paths = meta.paths
        self.distances = meta.distances
        self.cached_adj_loc_lists = meta.cached_adj_loc_lists
        self.angle_features = meta.angle_features
        self.angle_feat_size = meta.angle_feat_size
        self.image_feat_size = meta.image_feat_size

        self.init_simulator()

        self.random = random.Random(self.config.seed)

    def init(self, poses):
        self.sim.newEpisode(*list(zip(*poses)))
        return MatterportState(self)

    def init_simulator(self):
        self.sim_batch_size = self.config.trainer.batch_size
        self.sim = MatterSim.Simulator()
        self.sim.setRenderingEnabled(False)
        self.sim.setDiscretizedViewingAngles(True)
        self.sim.setCameraResolution(IMAGE_W, IMAGE_H)
        self.sim.setCameraVFOV(math.radians(VFOV))
        self.sim.setNavGraphPath(
            os.path.join(self.config.data_dir, 'connectivity'))
        self.sim.setBatchSize(self.sim_batch_size)
        self.sim.initialize()

    def get_shortest_path(self, scan, start, end):
        return self.paths[scan][start][end]

    def get_weighted_distance(self, scan, start, end):
        return self.distances[scan][start][end]

    def get_unweighted_distance(self, scan, start, end):
        return len(self.get_shortest_path(scan, start, end)) - 1

    def get_path_length(self, scan, path):
        length = 0
        for i, v in enumerate(path[1:]):
            u = path[i]
            length += self.get_weighted_distance(scan, u, v)
        return length


class MatterportState(object):

    def __init__(self, world):

        self.world = world

        self.states = []

        for sim_state in world.sim.getState():

            state = util.Struct()
            state.scan      = sim_state.scanId
            state.viewpoint = sim_state.location.viewpointId
            state.view_id   = sim_state.viewIndex
            state.heading   = sim_state.heading
            state.elevation = sim_state.elevation
            state.location  = sim_state.location

            long_id = '_'.join(
                [state.scan, state.viewpoint, str(state.view_id % 12)])
            state.adj_loc_list = world.cached_adj_loc_lists[long_id]

            state.curr_view_features = world.featurizer.get_features(
                state.scan, state.viewpoint)

            state.action_embeddings = util.build_action_embeddings(
                state.adj_loc_list, state.curr_view_features, world.angle_feat_size)

            state.curr_view_features = np.concatenate(
                (state.curr_view_features, world.angle_features[state.view_id]), -1)

            self.states.append(state)

    def __getitem__(self, i):
        return self.states[i]

    def __iter__(self):
        return iter(self.states)

    def __len__(self):
        return len(self.states)

    def step(self, actions):

        navigable_view_indices = []
        next_viewpoint_ids = []
        next_view_indices = []

        for i, (state, action) in enumerate(zip(self.states, actions)):
            loc = state.adj_loc_list[action]
            next_viewpoint_ids.append(loc['nextViewpointId'])
            next_view_indices.append(loc['absViewIndex'])
            navigable_view_indices.append(loc['absViewIndex'])

        new_states = self._navigate_to_locations(next_viewpoint_ids,
            next_view_indices, navigable_view_indices)

        return new_states

    def _navigate_to_locations(self,
            next_viewpoint_ids, next_view_indices, navigable_view_indices):

        sim = self.world.sim

        # Rotate to the view index assigned to the next viewpoint
        heading_deltas, elevation_deltas = util.calculate_headings_and_elevations_for_views(
            sim, navigable_view_indices)

        sim.makeAction(
            [0] * len(heading_deltas), heading_deltas, elevation_deltas)

        states = sim.getState()
        locationIds = []

        assert len(states) == len(next_viewpoint_ids) == len(navigable_view_indices)

        zipped_info = zip(states,
                          next_viewpoint_ids,
                          navigable_view_indices)

        for i, (state, next_viewpoint_id, navigable_view_index) in enumerate(zipped_info):

            # Check if rotation was done right
            assert state.viewIndex == navigable_view_index

            # Find index of the next viewpoint
            index = None
            for i, loc in enumerate(state.navigableLocations):
                if loc.viewpointId == next_viewpoint_id:
                    index = i
                    break
            assert index is not None, state.scanId + ' ' + state.location.viewpointId + ' ' + next_viewpoint_id
            locationIds.append(index)

        # Rotate to the target view index
        heading_deltas, elevation_deltas = util.calculate_headings_and_elevations_for_views(
            sim, next_view_indices)

        sim.makeAction(locationIds, heading_deltas, elevation_deltas)

        # Final check

        states = sim.getState()
        zipped_info = zip(states, next_viewpoint_ids, next_view_indices)
        for state, next_viewpoint_id, next_view_index in zipped_info:
            assert state.viewIndex == next_view_index
            assert state.location.viewpointId == next_viewpoint_id

        return MatterportState(self.world)


