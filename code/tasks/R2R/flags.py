import sys
sys.path.append('../../build')
import jsonargparse
import yaml
import numpy as np

from misc.util import Struct


parser = jsonargparse.ArgumentParser()


def update_config(source, target):
    for k in source.keys():
        if isinstance(source[k], dict):
            if k not in target:
                target[k] = {}
            update_config(source[k], target[k])
        elif source[k] is not None:
            target[k] = source[k]


def make_config():

    parser.add_argument('-config_file', type=str, help='Name of configuration file')

    parser.add_argument('-seed', type=int, help='Random seed')
    parser.add_argument('-name', type=str, help='Name of the experiment')
    parser.add_argument('-device_id', type=int, help='Which GPU to use (0, 1, 2, ...)')
    parser.add_argument('-data_dir', type=str, default='../../../data')

    parser.add_argument('-trainer.name', type=str, help='Name of the trainer class')
    parser.add_argument('-trainer.max_timesteps', type=int, help='Number of rollout steps')
    parser.add_argument('-trainer.max_iters', type=int, help='Number of training iterations')
    parser.add_argument('-trainer.log_every', type=int, help='Evaluate after every this number of iterations')
    parser.add_argument('-trainer.batch_size', type=int, help='Batch size')
    parser.add_argument('-trainer.save_every', type=int, help='Save the lastest model after every this number of iterations')

    parser.add_argument('-evaluate.log_name', type=str, default='eval')

    parser.add_argument('-nav_agent.model.vlnbert', type=str, help='oscar or prevalent')
    parser.add_argument('-nav_agent.model.max_instruction_length', type=int, help="Maximum length of input instructions")
    parser.add_argument('-nav_agent.model.img_feat_size', type=int, help='Number of view image features')
    parser.add_argument("-nav_agent.model.load_from", help='path of the trained model')
    parser.add_argument('-nav_agent.model.dropout', type=float)
    parser.add_argument('-nav_agent.model.featdropout', type=float)
    parser.add_argument('-nav_agent.model.lr', type=float, help="Learning rate")
    parser.add_argument('-nav_agent.model.epsilon', type=float, default=0.1)
    parser.add_argument("-nav_agent.model.angle_feat_size", type=int)


    flags = parser.parse_args()

    with open(flags.config_file) as f:
        config = yaml.safe_load(f)

    update_config(jsonargparse.namespace_to_dict(flags), config)

    config = Struct(**config)

    return config


if __name__ == '__main__':
    config = make_config()
    print(config)
