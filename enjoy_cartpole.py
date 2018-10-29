import gym
import sys

from reinforce.agent import AgentReinforce
from reinforce.enjoy_reinforce import enjoy

from misc.utils import CARTPOLE_ENV, CURRENT_EXPERIMENT, get_experiment_name


def main():
    env_id = CARTPOLE_ENV
    env = gym.make(env_id)
    params = AgentReinforce.Params(get_experiment_name(env_id, CURRENT_EXPERIMENT))
    return enjoy(env, params, deterministic=True)


if __name__ == '__main__':
    sys.exit(main())
