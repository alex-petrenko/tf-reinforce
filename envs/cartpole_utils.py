import gym

from misc.utils import CARTPOLE_ENV


class CartpoleWrapper(gym.core.Wrapper):
    """
    This wrapper adds 'reached_goal' property, where goal is to keep the cartpole upright for 200 steps.

    """
    def __init__(self, env):
        super(CartpoleWrapper, self).__init__(env)
        self._steps = 0
        self._max_steps = env.spec.max_episode_steps

    def reset(self):
        self._steps = 0
        return self.env.reset()

    def step(self, action):
        self._steps += 1
        return self.env.step(action)

    @property
    def reached_goal(self):
        return self._steps >= self._max_steps


def make_cartpole_env():
    env_id = CARTPOLE_ENV
    env = CartpoleWrapper(gym.make(env_id))
    env.seed(1)
    return env, env_id
