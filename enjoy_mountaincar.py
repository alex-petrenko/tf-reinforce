import sys

from reinforce.agent import AgentReinforce
from reinforce.enjoy_reinforce import enjoy

from envs.mountaincar_utils import make_mountaincar_env

from misc.utils import CURRENT_EXPERIMENT, get_experiment_name


def main():
    env, env_id = make_mountaincar_env()
    # experiment = get_experiment_name(env_id, CURRENT_EXPERIMENT)
    experiment = 'MountainCar-v0-reinforce_v039'
    params = AgentReinforce.Params(experiment)
    return enjoy(env, params, deterministic=False)


if __name__ == '__main__':
    sys.exit(main())
