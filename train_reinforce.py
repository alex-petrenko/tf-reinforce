import gym
import sys

from agent import AgentReinforce

from utils import CURRENT_ENV, CURRENT_EXPERIMENT, get_experiment_name


def train(env_id, params):
    env = gym.make(env_id)

    agent = AgentReinforce(env, params)
    agent.initialize()

    agent.learn(env)

    env.close()
    return 0


def main():
    """Script entry point."""
    env_id = CURRENT_ENV
    params = AgentReinforce.Params(get_experiment_name(env_id, CURRENT_EXPERIMENT))
    return train(env_id, params)


if __name__ == '__main__':
    sys.exit(main())
