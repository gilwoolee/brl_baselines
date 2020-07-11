class PSRLAgent(object):
    def __init__(self, learner):
        self.learner = learner

    def get_policy(self, env_sampler):
        # Sample an MDP from the distribution
        env = env_sampler.sample()
        policy = self.learner.learn(env)

        return policy
