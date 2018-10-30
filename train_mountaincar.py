import sys

from reinforce.agent import AgentReinforce
from reinforce.train_reinforce import train

from envs.mountaincar_utils import make_mountaincar_env
from misc.utils import CURRENT_EXPERIMENT, get_experiment_name


def main():
    """Script entry point."""
    env, env_id = make_mountaincar_env()
    custom_experiment = 'MountainCar-v0-reinforce_v043_test'
    experiment = get_experiment_name(env_id, CURRENT_EXPERIMENT) if custom_experiment is None else custom_experiment
    params = AgentReinforce.Params(experiment)
    params.learning_rate = 5e-5
    params.repeat_action = 4
    params.stats_episodes = 100
    return train(env, params)


if __name__ == '__main__':
    sys.exit(main())
