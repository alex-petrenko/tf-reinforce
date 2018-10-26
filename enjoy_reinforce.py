import gym
import sys

from agent import AgentReinforce

from utils import log, CURRENT_ENV, CURRENT_EXPERIMENT, get_experiment_name


def run_policy_loop(agent, env, max_num_episodes, deterministic=False):
    """Execute the policy and render onto the screen, using the standard agent interface."""
    agent.initialize()

    episode_rewards = []
    for _ in range(max_num_episodes):
        obs, done = env.reset(), False
        episode_reward = 0

        while not done:
            env.render()
            action = agent.best_action(obs, deterministic=deterministic)
            obs, rew, done, _ = env.step(action)
            episode_reward += rew

        env.render()

        episode_rewards.append(episode_reward)
        last_episodes = episode_rewards[-100:]
        avg_reward = sum(last_episodes) / len(last_episodes)
        log.info(
            'Episode reward: %f, avg reward for %d episodes: %f', episode_reward, len(last_episodes), avg_reward,
        )

    agent.finalize()
    env.close()
    return 0


def enjoy(env_id, params, max_num_episodes=1000000):
    env = gym.make(env_id)
    env.seed(0)

    agent = AgentReinforce(env, params)
    return run_policy_loop(agent, env, max_num_episodes)


def main():
    env_id = CURRENT_ENV
    params = AgentReinforce.Params(get_experiment_name(env_id, CURRENT_EXPERIMENT))
    return enjoy(env_id, params)


if __name__ == '__main__':
    sys.exit(main())
