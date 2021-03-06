import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

from os.path import join

from misc.utils import summaries_dir, experiments_dir, log


COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'black', 'purple', 'pink',
          'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
          'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue', 'yellow']


class Experiment:
    def __init__(self, name, descr):
        self.name = name
        self.descr = descr


class Plot:
    def __init__(self, name, axis, descr):
        self.name = name
        self.axis = axis
        self.descr = descr


def main():
    """Script entry point."""
    experiments = [
        Experiment('MountainCar-v0-reinforce_v047_linear_1e-3', 'Linear model, lr=1e-3'),
        Experiment('MountainCar-v0-reinforce_v047_linear_5e-4', 'Linear model, lr=5e-4'),
        Experiment('MountainCar-v0-reinforce_v047_mlp', 'MLP model, lr=5e-4'),
        Experiment('MountainCar-v0-reinforce_v047_mlp_1e-4', 'MLP model, lr=1e-4'),
    ]

    plots = [
        Plot('reinforce_aux_summary/avg_reward', 'average reward', 'Avg. reward for the last 100 episodes'),
        Plot(
            'reinforce_aux_summary/avg_success',
            'average success',
            'Avg. success (goal reached) for the last 100 episodes',
        ),
        Plot(
            'reinforce_agent_summary/policy_entropy',
            'policy entropy, nats',
            'Stochastic policy entropy',
        ),
    ]

    stop_at = 32000

    for plot in plots:
        fig = plt.figure(figsize=(10, 7))
        fig.add_subplot()

        for ex_i, experiment in enumerate(experiments):
            path_to_events_dir = summaries_dir(experiment.name)
            events_file = None
            for f in os.listdir(path_to_events_dir):
                if f.startswith('events'):
                    events_file = join(path_to_events_dir, f)
                    break

            if events_file is None:
                log.error('No events file for %s', experiment)
                continue

            steps, values = [], []

            for e in tf.train.summary_iterator(events_file):
                for v in e.summary.value:
                    if e.step >= stop_at:
                        break

                    if v.tag == plot.name:
                        steps.append(e.step)
                        values.append(v.simple_value)

            plt.plot(steps, values, color=COLORS[ex_i], label=experiment.descr)

        plt.xlabel('training batches (size=512)')
        plt.ylabel(plot.axis)
        plt.title(plot.descr)
        plt.grid(True)
        plt.legend()
        plt.savefig(join(experiments_dir(), 'mountaincar_{}.png'.format(plot.name.replace('/', '_'))))
        plt.close()

    return 0


if __name__ == '__main__':
    sys.exit(main())
