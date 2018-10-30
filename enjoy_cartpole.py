import sys

from reinforce.agent import AgentReinforce
from reinforce.enjoy_reinforce import enjoy

from envs.cartpole_utils import make_cartpole_env
from misc.utils import CURRENT_EXPERIMENT, get_experiment_name


def main():
    env, env_id = make_cartpole_env()
    custom_experiment = 'CartPole-v0-reinforce_v040_lr'
    experiment = get_experiment_name(env_id, CURRENT_EXPERIMENT) if custom_experiment is None else custom_experiment
    params = AgentReinforce.Params(experiment)
    return enjoy(env, params, deterministic=False)


if __name__ == '__main__':
    sys.exit(main())
