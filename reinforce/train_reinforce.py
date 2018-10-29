from reinforce.agent import AgentReinforce


def train(env, params):
    agent = AgentReinforce(env, params)
    agent.initialize()

    agent.learn(env)

    env.close()
    return 0
