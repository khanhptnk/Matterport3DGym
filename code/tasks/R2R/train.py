import os
import sys
import time
import logging
import numpy as np
from datetime import datetime

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
        trainer.train(datasets, env, agent, eval_splits=['val_seen', 'val_unseen'])

def configure():

    config = flags.make_config()

    config.command_line = 'python3 -u ' + ' '.join(sys.argv)

    config.data_dir = os.getenv('PT_DATA_DIR', config.data_dir)
    output_dir = os.getenv('PT_OUTPUT_DIR', 'experiments')
    config.experiment_dir = "%s/%s" % (output_dir, config.name)

    assert not os.path.exists(config.experiment_dir), \
            "Experiment %s already exists!" % config.experiment_dir
    os.makedirs(config.experiment_dir)

    torch.manual_seed(config.seed)
    random = np.random.RandomState(config.seed)
    config.random = random

    config.device = torch.device('cuda', config.device_id)

    config.start_time = time.time()

    log_file = os.path.join(config.experiment_dir, 'run.log')
    util.config_logging(log_file)
    logging.info(str(datetime.now()))
    logging.info(config.command_line)
    logging.info('Write log to %s' % log_file)
    logging.info(str(config))

    return config

if __name__ == '__main__':
    main()
