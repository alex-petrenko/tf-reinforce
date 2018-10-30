import gym

from misc.utils import MOUNTAINCAR_ENV


class MountaincarWrapper(gym.core.Wrapper):
    """
    REINFORCE struggles to solve Mountaincar without at least minor reward shaping, mainly because reward is very
    sparse in the beginning of the training.
    Here we modify the reward function by adding small reward for moving fast in any direction. This helps a lot.
    Repeating actions helps too.

    """
    def __init__(self, env):
        super(MountaincarWrapper, self).__init__(env)
        self._done = False
        self._steps = 0
        self._max_steps = env.spec.max_episode_steps

    def reset(self):
        self._done = False
        self._steps = 0
        return self.env.reset()

    @staticmethod
    def _shape_reward(reward, observation):
        v = observation[1]  # velocity
        reward += abs(v)
        return reward

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._done = done
        self._steps += 1
        reward = self._shape_reward(reward, observation)
        return observation, reward, done, info

    @property
    def reached_goal(self):
        return self._done and self._steps < self._max_steps


def make_mountaincar_env():
    env_id = MOUNTAINCAR_ENV
    return MountaincarWrapper(gym.make(env_id)), env_id
