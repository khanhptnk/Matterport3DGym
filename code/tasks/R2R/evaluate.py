import os
import sys
import time
import logging
import numpy as np
from datetime import datetime
from collections import defaultdict

import torch

import flags
import datacode
import environments
import trainers
import agents

from misc import util


def main():

    config = configure()

    datasets = datacode.load(config)
    env = environments.load(config)
    trainer = trainers.load(config)
    agent = agents.load(config)

    with torch.cuda.device(config.device_id):
        for eval_split in ['val_seen', 'val_unseen', 'test']:
            trainer.evaluate(eval_split, datasets[eval_split], env, agent,
                pred_save_name=config.evaluate.log_name)

def configure():

    config = flags.make_config()

    config.command_line = 'python3 -u ' + ' '.join(sys.argv)

    config.name = config.nav_agent.model.load_from.split('/')[-2]

    config.experiment_dir = os.path.join("experiments/%s" % config.name)
    assert os.path.exists(config.experiment_dir), \
            "Experiment %s not exists!" % config.experiment_dir

    torch.manual_seed(config.seed)
    random = np.random.RandomState(config.seed)
    config.random = random

    config.device = torch.device('cuda', config.device_id)

    config.start_time = time.time()

    log_file = os.path.join(config.experiment_dir, config.evaluate.log_name + '.log')
    util.config_logging(log_file)
    logging.info(str(datetime.now()))
    logging.info(config.command_line)
    logging.info(str(config))

    return config

if __name__ == '__main__':
    main()
