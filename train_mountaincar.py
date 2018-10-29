import sys

from reinforce.agent import AgentReinforce
from reinforce.train_reinforce import train

from envs.mountaincar_utils import make_mountaincar_env
from misc.utils import CURRENT_EXPERIMENT, get_experiment_name


def main():
    """Script entry point."""
    env, env_id = make_mountaincar_env()
    params = AgentReinforce.Params(get_experiment_name(env_id, CURRENT_EXPERIMENT))
    return train(env, params)


if __name__ == '__main__':
    sys.exit(main())
