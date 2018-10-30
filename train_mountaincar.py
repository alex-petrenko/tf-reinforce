import sys

from reinforce.agent import AgentReinforce
from reinforce.train_reinforce import train

from envs.mountaincar_utils import make_mountaincar_env
from misc.utils import CURRENT_EXPERIMENT, get_experiment_name, parse_args


def main():
    """Script entry point."""
    args = parse_args()
    experiment = args.experiment

    env, env_id = make_mountaincar_env()
    if experiment is None:
        experiment = get_experiment_name(env_id, CURRENT_EXPERIMENT)

    params = AgentReinforce.Params(experiment)
    params.learning_rate = 1e-4
    params.repeat_action = 4

    return train(env, params)


if __name__ == '__main__':
    sys.exit(main())
