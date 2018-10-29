import time

from reinforce.agent import AgentReinforce

from misc.utils import log


def run_policy_loop(agent, env, max_num_episodes, deterministic=False):
    """Execute the policy and render onto the screen, using the standard agent interface."""
    fps = env.metadata.get('video.frames_per_second')

    agent.initialize()

    episode_rewards = []
    for i in range(max_num_episodes):
        obs, done = env.reset(), False
        episode_reward = episode_len = 0

        while not done:
            env.render()
            time.sleep(1 / fps)
            action = agent.best_action(obs, deterministic=deterministic)
            obs, rew, done, _ = env.step(action)
            episode_len += 1
            episode_reward += rew

        env.render()
        time.sleep(2 * (1 / fps))

        episode_rewards.append(episode_reward)
        last_episodes = episode_rewards[-100:]
        avg_reward = sum(last_episodes) / len(last_episodes)
        log.info(
            'Episode reward: %f, avg reward for %d episodes: %f', episode_reward, len(last_episodes), avg_reward,
        )

    agent.finalize()
    env.close()
    return 0


def enjoy(env, params, max_num_episodes=1000000, deterministic=False):
    env.seed(0)
    agent = AgentReinforce(env, params)
    return run_policy_loop(agent, env, max_num_episodes, deterministic)
