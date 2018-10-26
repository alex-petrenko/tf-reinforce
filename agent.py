import numpy as np
import tensorflow as tf

from utils import log, model_dir, summaries_dir


EPS = 1e-9


class CategoricalPd:
    def __init__(self, logits):
        self.logits = logits

    def neglogp(self, x):
        x = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        return tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=x)

    def entropy(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0 + EPS) - a0), axis=-1)

    def sample(self):
        u = tf.random_uniform(tf.shape(self.logits), dtype=self.logits.dtype)
        return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)


class StochasticLinearPolicy:
    def __init__(self, action_space, observation_shape):
        self.x = tf.placeholder(tf.float32, [None] + list(observation_shape), 'observation')

        num_actions = action_space.n

        hidden = tf.layers.dense(self.x, 64, tf.nn.relu)
        self.action_logits = tf.layers.dense(hidden, num_actions, activation=None)

        # w = tf.get_variable('w', [observation_shape[0], num_actions], tf.float32)
        # b = tf.get_variable('b', [num_actions])
        #
        # logits = tf.matmul(self.x, w) + b

        self.action_distribution = CategoricalPd(self.action_logits)
        self.best_action_deterministic = tf.argmax(self.action_logits, axis=1)
        self.act = self.action_distribution.sample()


class AgentReinforce:
    class Params:
        def __init__(self, experiment_name):
            self.experiment_name = experiment_name

            self.e_greedy = 0.05

            self.use_gpu = False

            self.learning_rate = 1e-3
            self.train_for = 10000

            self.print_every = 100
            self.save_every = 100
            self.summaries_every = 100

    def __init__(self, env, params):
        self.params = params

        self.best_avg_reward = self.best_saved_avg_reward = -1000.0

        self.session = None
        tf.reset_default_graph()

        global_step = tf.train.get_or_create_global_step()

        action_space = env.action_space
        observation_shape = env.observation_space.shape
        self.policy = StochasticLinearPolicy(action_space, observation_shape)

        self.selected_actions = tf.placeholder(tf.int32, [None])
        self.episode_reward = tf.placeholder(tf.float32, [])

        self.neglogp_actions = self.policy.action_distribution.neglogp(self.selected_actions)

        # maximize probabilities of actions that give high advantage
        loss = tf.reduce_mean(self.neglogp_actions * self.episode_reward)

        # training
        self.train = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=global_step,
            learning_rate=self.params.learning_rate,
            optimizer=tf.train.AdamOptimizer,
            clip_gradients=20.0,
        )

        # summaries for the agent and the training process
        with tf.name_scope('reinforce_agent_summary'):
            tf.summary.histogram('actions', self.policy.action_logits)
            tf.summary.scalar('action_avg', tf.reduce_mean(tf.to_float(self.policy.act)))

            tf.summary.histogram('selected_actions', self.selected_actions)
            tf.summary.scalar('selected_action_avg', tf.reduce_mean(tf.to_float(self.selected_actions)))

            tf.summary.scalar('policy_entropy', tf.reduce_mean(self.policy.action_distribution.entropy()))

            tf.summary.scalar('loss', loss)

            summary_dir = summaries_dir(self.params.experiment_name)
            self.summary_writer = tf.summary.FileWriter(summary_dir)

            self.summaries = tf.summary.merge_all()

        with tf.name_scope('reinforce_aux_summary'):
            self.avg_reward_placeholder = tf.placeholder(tf.float32, [])
            self.avg_length_placeholder = tf.placeholder(tf.float32, [])
            tf.summary.scalar('avg_reward', self.avg_reward_placeholder, collections=['aux'])
            tf.summary.scalar('avg_lenght', self.avg_length_placeholder, collections=['aux'])
            self.aux_summaries = tf.summary.merge_all(key='aux')

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
        if next_step % self.params.save_every == 0 and avg_reward > self.best_avg_reward:
            log.info('Step #%d, saving model with reward %.3f...', step, avg_reward)
            self.best_saved_avg_reward = avg_reward
            saver_path = model_dir(self.params.experiment_name) + '/' + self.__class__.__name__
            self.saver.save(self.session, saver_path, global_step=step)

    def _maybe_print(self, step, avg_length, avg_reward):
        if step % self.params.print_every == 0:
            log.info('<====== Step %d ======>', step)
            log.info('Avg. 100 episode lenght: %.3f', avg_length)
            if avg_reward > self.best_avg_reward:
                self.best_avg_reward = avg_reward
                log.info('<<<<< New record! %.3f >>>>>\n', self.best_avg_reward)
            log.info('Avg. 100 episode reward: %.3f (best: %.3f)', avg_reward, self.best_avg_reward)

    def _get_action(self, observation, deterministic=False):
        actions_batch = self.session.run(
            self.policy.best_action_deterministic if deterministic else self.policy.act,
            feed_dict={self.policy.x: [observation]}
        )
        return actions_batch[0]

    def _epsilon_greedy(self, action, env):
        if np.random.random() < self.params.e_greedy:
            return np.random.randint(env.action_space.n)
        return action

    def _train_step(self, step, observations, actions, episode_reward):
        with_summaries = (step % self.params.summaries_every == 0)  # prevent summaries folder from growing too large
        summaries = [self.summaries] if with_summaries else []

        result = self.session.run(
            [self.train] + summaries,
            feed_dict={
                self.policy.x: observations,
                self.selected_actions: actions,
                self.episode_reward: episode_reward,
            },
        )

        step = tf.train.global_step(self.session, tf.train.get_global_step())
        if with_summaries:
            summary = result[-1]
            self.summary_writer.add_summary(summary, global_step=step)

        return step

    def _maybe_aux_summaries(self, step, avg_reward, avg_length):
        if step % self.params.summaries_every == 0:
            summary = self.session.run(
                self.aux_summaries,
                feed_dict={
                    self.avg_reward_placeholder: avg_reward,
                    self.avg_length_placeholder: avg_length,
                },
            )
            self.summary_writer.add_summary(summary, global_step=step)

    @staticmethod
    def _modify_reward_position(x):
        # adjust reward based on car position
        reward = x + 0.5

        # adjust reward for task completion
        if x >= 0.5:
            reward += 1

        return reward

    def learn(self, env):
        step = tf.train.global_step(self.session, tf.train.get_global_step())

        episode_lengths = []
        episode_rewards = []

        def update_hist_buffer(buffer, value):
            buffer.append(value)
            while len(buffer) > 100:
                del buffer[0]

        while step < self.params.train_for:
            observation = env.reset()

            observations = []
            actions = []

            episode_len = 0
            episode_reward = 0
            done = False
            while not done:
                action = self._get_action(observation)
                action = self._epsilon_greedy(action, env)

                observation, reward, done, _ = env.step(action)
                episode_len += 1

                x = observation[0]
                reward = self._modify_reward_position(x)

                if done and episode_len < 200:
                    log.info('Solved in %d steps!', episode_len)

                observations.append(observation)
                actions.append(action)

                episode_reward += reward

            update_hist_buffer(episode_lengths, episode_len)
            update_hist_buffer(episode_rewards, episode_reward)

            avg_length = np.mean(episode_lengths)
            avg_reward = np.mean(episode_rewards)

            step = self._train_step(step, observations, actions, episode_reward)
            self._maybe_save(step, avg_reward)
            self._maybe_print(step, avg_length, avg_reward)
            self._maybe_aux_summaries(step, avg_reward, avg_length)
