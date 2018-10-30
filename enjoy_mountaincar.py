import sys

from reinforce.agent import AgentReinforce
from reinforce.enjoy_reinforce import enjoy

from envs.mountaincar_utils import make_mountaincar_env

from misc.utils import CURRENT_EXPERIMENT, get_experiment_name, parse_args


def main():
    args = parse_args()
    experiment = args.experiment

    env, env_id = make_mountaincar_env()
    if experiment is None:
        experiment = get_experiment_name(env_id, CURRENT_EXPERIMENT)

    params = AgentReinforce.Params(experiment)
    return enjoy(env, params, deterministic=False)


if __name__ == '__main__':
    sys.exit(main())
