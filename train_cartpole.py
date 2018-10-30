import sys

from reinforce.agent import AgentReinforce
from reinforce.train_reinforce import train

from envs.cartpole_utils import make_cartpole_env
from misc.utils import CURRENT_EXPERIMENT, get_experiment_name


def main():
    """Script entry point."""
    env, env_id = make_cartpole_env()
    params = AgentReinforce.Params(get_experiment_name(env_id, CURRENT_EXPERIMENT))

    # no additional exploration required, policy stochasticity is enough for cartpole
    params.initial_e_greedy = 0.2
    params.min_e_greedy = 0.00
    params.gamma = 0.95
    params.learning_rate = 1e-4
    params.min_batch_size = 1  # do training step after every episode

    return train(env, params)


if __name__ == '__main__':
    sys.exit(main())
