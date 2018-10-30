import sys
import time

import numpy as np
import tensorflow as tf

from functools import partial

from misc.utils import log, model_dir, summaries_dir


EPS = 1e-9


class CategoricalPd:
    def __init__(self, logits):
        self.logits = logits
        self.prob = tf.nn.softmax(self.logits)

    def neglogp(self, x):
        x = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        return tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=x)

    def entropy(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0 + EPS) - a0), axis=-1)


class StochasticPolicy:
    def __init__(self, action_space, observation_shape, model):
        self.x = tf.placeholder(tf.float32, [None] + list(observation_shape), 'observation')
        num_actions = action_space.n

        if model == 'linear':
            hidden = tf.layers.dense(self.x, 64, activation=None)
            self.action_logits = tf.layers.dense(hidden, num_actions, activation=None)
        elif model == 'mlp':
            hidden = tf.layers.dense(
                self.x, 64, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1),
            )
            hidden = tf.layers.dense(
                hidden, 64, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1),
            )
            self.action_logits = tf.layers.dense(
                hidden, num_actions, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1),
            )
        else:
            raise Exception('Unknown model')

        self.weights = tf.get_default_graph().get_tensor_by_name(self.action_logits.name.split('/')[0] + '/kernel:0')
        self.biases = tf.get_default_graph().get_tensor_by_name(self.action_logits.name.split('/')[0] + '/bias:0')

        self.action_distribution = CategoricalPd(self.action_logits)
        self.best_action_deterministic = tf.argmax(self.action_logits, axis=1)

        self.timesteps = tf.placeholder(tf.float32, [None], 'timesteps')
        timesteps = tf.reshape(self.timesteps, [-1, 1])
        log_timesteps = tf.log(timesteps + 1.0)
        value_input = tf.concat([hidden, log_timesteps], axis=1)
        value_h = tf.layers.dense(value_input, 64, activation=tf.nn.relu)
        value = tf.layers.dense(value_h, 1, activation=None)
        self.value = tf.squeeze(value, axis=[1])


class AgentReinforce:
    class Params:
        def __init__(self, experiment_name):
            self.experiment_name = experiment_name

            self.repeat_action = 1

            self.gamma = 0.99
            self.initial_e_greedy = 0.5
            self.min_e_greedy = 0.1
            self.model = 'mlp'

            self.use_gpu = False

            self.learning_rate = 2e-4
            self.min_batch_size = 512
            self.train_for = 1000000

            self.stats_episodes = 100
            self.print_every = 100
            self.save_every = 100
            self.summaries_every = 100

        def log(self):
            log.info('Algorithm parameters:')
            for key, value in self.__dict__.items():
                log.info('%s: %r', key, value)

    def __init__(self, env, params):
        self.params = params

        self.session = None
        tf.reset_default_graph()

        global_step = tf.train.get_or_create_global_step()

        # store important statistic in a session to save/load it when needed
        self.avg_reward_placeholder = tf.placeholder(tf.float32, [], 'new_avg_reward')
        self.best_avg_reward = tf.Variable(-sys.float_info.max, dtype=tf.float32)
        self.best_saved_reward = tf.Variable(-sys.float_info.max, dtype=tf.float32)
        self.success_rate_placeholder = tf.placeholder(tf.float32, [], 'new_success_rate')
        self.best_success_rate = tf.Variable(0.0, dtype=tf.float32)

        def update_best_value(best_value, new_value):
            return tf.assign(best_value, tf.maximum(new_value, best_value))
        self.update_best_reward = update_best_value(self.best_avg_reward, self.avg_reward_placeholder)
        self.update_best_saved_reward = update_best_value(self.best_saved_reward, self.avg_reward_placeholder)
        self.update_success_rate = update_best_value(self.best_success_rate, self.success_rate_placeholder)

        self.e_greedy_coeff = tf.train.exponential_decay(
            self.params.initial_e_greedy, tf.cast(global_step, tf.float32), 50.0, 0.95, staircase=True,
        )
        self.e_greedy_coeff = tf.maximum(self.e_greedy_coeff, self.params.min_e_greedy)

        action_space = env.action_space
        observation_shape = env.observation_space.shape

        self.policy = StochasticPolicy(action_space, observation_shape, self.params.model)

        self.selected_actions = tf.placeholder(tf.int32, [None])
        self.discounted_rewards = tf.placeholder(tf.float32, [None])

        self.neglogp_actions = self.policy.action_distribution.neglogp(self.selected_actions)

        # maximize probabilities of actions that give high advantage
        action_loss = tf.reduce_mean(self.neglogp_actions * self.discounted_rewards)

        # optimize the value estimation for better baseline
        value_loss = tf.losses.absolute_difference(self.policy.value, self.discounted_rewards)

        loss = action_loss  # + value_loss

        self.train = tf.train.AdamOptimizer(self.params.learning_rate).minimize(loss, global_step=global_step)

        # summaries for the agent and the training process
        with tf.name_scope('reinforce_agent_summary'):
            tf.summary.histogram('actions', self.policy.action_logits)
            tf.summary.histogram('actions_prob', self.policy.action_distribution.prob)

            tf.summary.histogram('selected_actions', self.selected_actions)
            tf.summary.scalar('selected_action_avg', tf.reduce_mean(tf.to_float(self.selected_actions)))

            tf.summary.scalar('policy_entropy', tf.reduce_mean(self.policy.action_distribution.entropy()))

            tf.summary.histogram('weights', self.policy.weights)
            tf.summary.histogram('biases', self.policy.biases)

            tf.summary.scalar('e_greedy', self.e_greedy_coeff)

            tf.summary.scalar('action_loss', action_loss)
            # tf.summary.scalar('value_loss', value_loss)

            summary_dir = summaries_dir(self.params.experiment_name)
            self.summary_writer = tf.summary.FileWriter(summary_dir)

            self.summaries = tf.summary.merge_all()

        with tf.name_scope('reinforce_aux_summary'):
            self.avg_length_placeholder = tf.placeholder(tf.float32, [])
            tf.summary.scalar('avg_reward', self.avg_reward_placeholder, collections=['aux'])
            tf.summary.scalar('avg_lenght', self.avg_length_placeholder, collections=['aux'])
            tf.summary.scalar('avg_success', self.success_rate_placeholder, collections=['aux'])
            self.aux_summaries = tf.summary.merge_all(key='aux')

        self.saved = False
        self.saver = tf.train.Saver(max_to_keep=3)

    def initialize(self):
        config = tf.ConfigProto(
            device_count={'GPU': 100 if self.params.use_gpu else 0},
            log_device_placement=False,
        )
        self.session = tf.Session(config=config)

        checkpoint_dir = model_dir(self.params.experiment_name)
        try:
            self.saver.restore(self.session, tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir))
            self.saved = True
        except ValueError:
            log.info('Didn\'t find a valid restore point, start from scratch')
            self.session.run(tf.global_variables_initializer())
        log.info('Initialized!')

    def finalize(self):
        self.session.close()

    def best_action(self, observation, deterministic):
        return self._get_action(observation, deterministic)

    def _maybe_save(self, step, avg_reward):
        next_step = step + 1
        if next_step % self.params.save_every != 0:
            return

        best_reward = self.best_saved_reward.eval(session=self.session)
        if avg_reward < best_reward and self.saved:
            return

        log.info('Step #%d, saving model with reward %.3f!', step, avg_reward)
        saver_path = model_dir(self.params.experiment_name) + '/' + self.__class__.__name__
        self.saver.save(self.session, saver_path, global_step=step)
        self.session.run(self.update_best_saved_reward, feed_dict={self.avg_reward_placeholder: avg_reward})
        self.saved = True

    def _maybe_print(self, step, avg_length, avg_reward, avg_success, avg_fps):
        if step % self.params.print_every == 0:
            log.info('<====== Step %d ======>', step)
            log.info('FPS: %.3f', avg_fps)
            log.info('Avg. episode lenght: %.3f', avg_length)
            log.info(
                'Avg. success rate: %.3f (best: %.3f)',
                avg_success, self.best_success_rate.eval(session=self.session),
            )
            log.info(
                'Avg. %d episode reward: %.3f (best: %.3f)',
                self.params.stats_episodes, avg_reward, self.best_avg_reward.eval(session=self.session),
            )

    def _get_action(self, observation, deterministic=False):
        if deterministic:
            return self.session.run(
                self.policy.best_action_deterministic,
                feed_dict={self.policy.x: [observation]},
            ).ravel()

        actions_prob = self.session.run(
            self.policy.action_distribution.prob,
            feed_dict={self.policy.x: [observation]},
        )

        actions_prob = np.ravel(actions_prob)
        action = np.random.choice(len(actions_prob), p=actions_prob)
        return action

    @staticmethod
    def _epsilon_greedy(action, env, e_greedy):
        if np.random.random() < e_greedy:
            return np.random.randint(env.action_space.n)
        return action

    def _calculate_baseline(self, observations, timesteps):
        return self.session.run(
            self.policy.value,
            feed_dict={
                self.policy.x: observations,
                self.policy.timesteps: np.array(timesteps).astype(np.float32),
            }
        )

    def _train_policy_step(self, step, observations, timesteps, actions, discounted_rewards):
        with_summaries = (step % self.params.summaries_every == 0)  # prevent summaries folder from growing too large
        summaries = [self.summaries] if with_summaries else []

        result = self.session.run(
            [self.train] + summaries,
            feed_dict={
                self.policy.x: observations,
                # self.policy.timesteps: np.array(timesteps).astype(np.float32),
                self.selected_actions: actions,
                self.discounted_rewards: discounted_rewards,
            },
        )

        step = tf.train.global_step(self.session, tf.train.get_global_step())
        if with_summaries:
            summary = result[-1]
            self.summary_writer.add_summary(summary, global_step=step)

        return step

    def _maybe_update_scoreboard(self, stats_episodes, avg_reward, avg_success_rate):
        if stats_episodes >= self.params.stats_episodes:
            if avg_reward > self.best_avg_reward.eval(session=self.session) + EPS:
                log.info('New best reward %.3f!', avg_reward)

            self.session.run(
                [self.update_best_reward, self.update_success_rate],
                feed_dict={
                    self.avg_reward_placeholder: avg_reward,
                    self.success_rate_placeholder: avg_success_rate,
                },
            )

    def _maybe_aux_summaries(self, step, avg_reward, avg_length, avg_success):
        if step % self.params.summaries_every == 0:
            summary = self.session.run(
                self.aux_summaries,
                feed_dict={
                    self.avg_reward_placeholder: avg_reward,
                    self.avg_length_placeholder: avg_length,
                    self.success_rate_placeholder: avg_success,
                },
            )
            self.summary_writer.add_summary(summary, global_step=step)

    def _discounted_rewards(self, rewards):
        total_reward = 0
        rewards_to_go = np.zeros_like(rewards)
        for i in reversed(range(len(rewards))):
            total_reward = rewards[i] + self.params.gamma * total_reward
            rewards_to_go[i] = total_reward
        return rewards_to_go

    def learn(self, env):
        log.info('Start training...')
        self.params.log()

        step = tf.train.global_step(self.session, tf.train.get_global_step())

        stats_episode_lengths, stats_rewards, stats_goal, stats_max_rewards = [], [], [], []

        def update_hist_buffer_len(buffer, value, max_len):
            buffer.append(value)
            while len(buffer) > max_len:
                del buffer[0]

        update_hist_buffer = partial(update_hist_buffer_len, max_len=self.params.stats_episodes)

        min_batch_size = self.params.min_batch_size
        observations, actions, rewards, timesteps = [], [], [], []

        while step < self.params.train_for:
            e_greedy_coeff = self.e_greedy_coeff.eval(session=self.session)

            observation = env.reset()
            episode_start = time.time()
            episode_obs, episode_rewards = [], []
            done = False
            while not done:
                action = self._get_action(observation)
                action = self._epsilon_greedy(action, env, e_greedy_coeff)

                for _ in range(self.params.repeat_action):
                    episode_obs.append(observation)
                    observation, reward, done, _ = env.step(action)
                    actions.append(action)
                    episode_rewards.append(reward)
                    if done:
                        break

            # episode finished
            episode_len = len(episode_rewards)

            observations.extend(episode_obs)
            timesteps.extend(np.arange(episode_len))

            episode_reward = sum(episode_rewards)
            goal_reached = False

            if hasattr(env, 'reached_goal'):
                goal_reached = env.reached_goal

            episode_sec = time.time() - episode_start
            avg_fps = episode_len / episode_sec

            update_hist_buffer(stats_episode_lengths, episode_len)
            update_hist_buffer(stats_rewards, episode_reward)
            update_hist_buffer(stats_goal, goal_reached)

            avg_length = np.mean(stats_episode_lengths)
            avg_reward = np.mean(stats_rewards)
            avg_success_rate = sum(stats_goal) / len(stats_goal)
            self._maybe_update_scoreboard(len(stats_rewards), avg_reward, avg_success_rate)

            # discounted_rewards = self._discounted_rewards(episode_rewards)
            # baseline = self._calculate_baseline(episode_obs, np.arange(episode_len))
            # discounted_rewards -= baseline

            baseline_reward = avg_reward / episode_len
            # subtract baseline reward element-wise
            episode_rewards_baseline = np.asarray(episode_rewards) - baseline_reward
            discounted_rewards = self._discounted_rewards(episode_rewards_baseline)

            # if self.params.baseline_reward:
            #     # baseline_reward = avg_reward / episode_len
            #
            #     discount = 1.0
            #     for i, r in enumerate(discounted_rewards):
            #         discounted_rewards[i] = discount * r
            #         discount *= self.params.gamma
            #
            #     # update_hist_buffer_len(baseline_rewards, discounted_rewards, self.params.baseline_reward_episodes)
            #     # subtract baseline reward element-wise
            #     # episode_rewards_baseline = np.asarray(episode_rewards) - baseline_reward
            # else:
            #     discounted_rewards -= np.mean(discounted_rewards)
            #     discounted_rewards /= np.std(discounted_rewards)

            rewards.extend(discounted_rewards)

            episode_max_abs_reward = np.max(np.abs(discounted_rewards))
            update_hist_buffer(stats_max_rewards, episode_max_abs_reward)
            assert len(rewards) == len(observations)

            while len(observations) >= min_batch_size:
                # enough data for training step
                step = self._train_policy_step(step, observations, timesteps, actions, rewards)
                self._maybe_save(step, avg_reward)
                self._maybe_print(step, avg_length, avg_reward, avg_success_rate, avg_fps)
                self._maybe_aux_summaries(step, avg_reward, avg_length, avg_success_rate)

                observations, actions, rewards, timesteps = [], [], [], []