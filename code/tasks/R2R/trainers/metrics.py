import os
import sys
import math


metric_list = ['dist', 'score', 'spl', 'ndtw', 'sdtw', 'path_len']

success_radius = 3

def get_metrics():
    return metric_list

def format_results(result_dict):

    result_strings = []
    result_strings.append('score %.1f' % (result_dict['score'] * 100))
    result_strings.append('spl %.1f'   % (result_dict['spl'] * 100))
    result_strings.append('dist %.2f'  % result_dict['dist'])
    result_strings.append('ndtw %.1f'  % (result_dict['ndtw'] * 100))
    result_strings.append('sdtw %.1f'  % (result_dict['sdtw'] * 100))
    result_strings.append('path_len %.1f' % result_dict['path_len'])

    return ', '.join(result_strings)

def init_value(metric_name):
    if metric_name in ['dist', 'path_len']:
        return 1e9
    if metric_name in ['score', 'spl', 'ndtw', 'sdtw']:
        return -1e9
    raise ValueError('%s is not a valid metric' % metric_name)

def is_better(metric_name, a, b):
    if metric_name in ['dist', 'path_len']:
        return a < b
    if metric_name in ['score', 'spl', 'ndtw', 'sdtw']:
        return a > b
    raise ValueError('%s is not a valid metric' % metric_name)

def eval(world, scan, pred_path, gold_path):

    pred_goal = pred_path[-1]
    gold_goal = gold_path[-1]

    dist = world.get_weighted_distance(scan, pred_goal, gold_goal)
    score = dist <= success_radius

    pred_travel_dist = world.get_path_length(scan, pred_path) + 1e-6
    gold_travel_dist = world.get_path_length(scan, gold_path) + 1e-6

    spl = score * gold_travel_dist / max(pred_travel_dist, gold_travel_dist)

    ndtw = compute_ndtw(world, scan, pred_path, gold_path)
    sdtw = compute_sdtw(world, scan, pred_path, gold_path)

    path_len = world.get_path_length(scan, pred_path)

    result = {
            'dist': dist,
            'score': score,
            'spl': spl,
            'ndtw': ndtw,
            'sdtw': sdtw,
            'path_len': path_len
        }

    return result

"""
def eval(world, scan, pred_path, gold_path, valid_goals):

    result = {}

    result['intended'] = eval_single_path(world, scan, pred_path, gold_path)
    result['intended']['path'] = gold_path

    best_dist, gold_path = None, None
    for goal in valid_goals:
        dist = world.get_weighted_distance(scan, pred_path[-1], goal)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            gold_path = world.get_shortest_path(scan, pred_path[0], goal)

    result['closest'] = eval_single_path(world, scan, pred_path, gold_path)
    result['closest']['path'] = gold_path

    return result
"""

def compute_ndtw(world, scan, pred_path, gold_path):
    r = gold_path
    q = pred_path
    c = [[1e9] * (len(q) + 1) for _ in range(len(r) + 1)]
    c[0][0] = 0

    for i in range(1, len(r) + 1):
        for j in range(1, len(q) + 1):
            d = world.get_weighted_distance(scan, r[i - 1], q[j - 1])
            c[i][j] = min(c[i - 1][j], c[i][j - 1], c[i - 1][j - 1]) + d

    return math.exp(-c[len(r)][len(q)] / (len(r) * success_radius))

def compute_sdtw(world, scan, pred_path, gold_path):
    d = world.get_weighted_distance(scan, pred_path[-1], gold_path[-1])
    if d > success_radius:
        return 0
    return compute_ndtw(world, scan, pred_path, gold_path)

