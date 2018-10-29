import gym

from misc.utils import MOUNTAINCAR_ENV


class MountaincarWrapper(gym.core.Wrapper):
    """
    Pretty much impossible to solve mountaincar with vanilla REINFORCE without some sort of reward shaping, unless
    some clever exploration tricks are used (beyond e-greedy).

    """
    def __init__(self, env):
        super(MountaincarWrapper, self).__init__(env)
        self._done = False

    def reset(self):
        self._done = False
        return self.env.reset()

    @staticmethod
    def _shape_reward(reward, observation):
        v = observation[1]  # velocity
        reward += abs(v)
        return reward

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._done = done
        reward = self._shape_reward(reward, observation)
        return observation, reward, done, info

    @property
    def reached_goal(self):
        return self._done and not self._past_limit()


def make_mountaincar_env():
    env_id = MOUNTAINCAR_ENV
    return MountaincarWrapper(gym.make(env_id)), env_id
