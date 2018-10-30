import os
import logging
import argparse

from os.path import join

from colorlog import ColoredFormatter


# Logging

def init_logger():
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    formatter = ColoredFormatter(
        '%(log_color)s[%(asctime)s] %(message)s',
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'white,bold',
            'WARNING':  'yellow',
            'ERROR':    'red,bold',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )
    handler.setFormatter(formatter)

    logger = logging.getLogger('reinforce')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []       # No duplicated handlers
    logger.propagate = False   # workaround for duplicated logs in ipython
    logger.addHandler(handler)
    return logger


log = init_logger()


# Filesystem helpers

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def project_root():
    """
    Keep models, parameters and summaries at the root of this project's directory tree.
    :return: full path to the root dir of this project.
    """
    return os.path.dirname(os.path.dirname(__file__))


def experiments_dir():
    return ensure_dir_exists(join(project_root(), '.experiments'))


def experiment_dir(experiment):
    return ensure_dir_exists(join(experiments_dir(), experiment))


def model_dir(experiment):
    return ensure_dir_exists(join(experiment_dir(experiment), '.model'))


def summaries_dir(experiment):
    return ensure_dir_exists(join(experiment_dir(experiment), '.summary'))


# Command line arguments

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # common args
    parser.add_argument('--experiment', type=str, default=None)
    args = parser.parse_args()
    for arg in vars(args):
        log.info('%s %r', arg, getattr(args, arg))
    return args


# Keeping track of experiments

def get_experiment_name(env_id, name):
    return '{}-{}'.format(env_id, name)


CURRENT_EXPERIMENT = 'reinforce_v001'
MOUNTAINCAR_ENV = 'MountainCar-v0'
CARTPOLE_ENV = 'CartPole-v0'
