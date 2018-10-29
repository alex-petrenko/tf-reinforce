import gym
import sys

from reinforce.agent import AgentReinforce

from reinforce.train_reinforce import train

from misc.utils import CARTPOLE_ENV, CURRENT_EXPERIMENT, get_experiment_name


def main():
    """Script entry point."""
    env_id = CARTPOLE_ENV

    params = AgentReinforce.Params(get_experiment_name(env_id, CURRENT_EXPERIMENT))
    params.batch_size = 256
    params.min_e_greedy = 0.05
    params.gamma = 0.99

    env = gym.make(env_id)
    return train(env, params)


if __name__ == '__main__':
    sys.exit(main())
