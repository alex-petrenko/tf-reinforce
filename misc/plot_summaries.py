import sys
import tensorflow as tf

from os.path import join

from misc.utils import summaries_dir


def main():
    """Script entry point."""
    experiment = 'MountainCar-v0-reinforce_v013'
    path_to_events_file = summaries_dir(experiment)
    path_to_events_file = join(path_to_events_file, 'events.out.tfevents.1540795053.alex-laptop')

    for e in tf.train.summary_iterator(path_to_events_file):
        for v in e.summary.value:
            if v.tag == 'reinforce_aux_summary/avg_reward':
                print(e.step)
                print(v)

    return 0


if __name__ == '__main__':
    sys.exit(main())
