from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from torch import nn
env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!



class ProjectAgent:
    def __init__(self, config, policy_network):
        self.device = "cuda" if next(policy_network.parameters()).is_cuda else "cpu"
        self.scalar_dtype = next(policy_network.parameters()).dtype
        self.policy = policy_network
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.99
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = torch.optim.Adam(list(self.policy.parameters()), lr=lr)
        self.nb_episodes = config['nb_episodes'] if 'nb_episodes' in config.keys() else 1

    def sample_action_and_log_prob(self, x):
        probabilities = self.policy(torch.as_tensor(x))
        action_distribution = Categorical(probabilities)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)
        return action.item(), log_prob

    def one_gradient_step(self, env):
        # run trajectories until done
        episodes_sum_of_rewards = []
        log_probs = []
        returns = []
        for ep in range(self.nb_episodes):
            x, _ = env.reset()
            rewards = []
            episode_cum_reward = 0
            while (True):
                x = np.log(x)
                a, log_prob = self.sample_action_and_log_prob(x)
                y, r, done, trunc, _ = env.step(a)
                log_probs.append(log_prob)
                rewards.append(r)
                episode_cum_reward += r
                x = y
                if done or trunc:
                    # compute returns-to-go
                    new_returns = []
                    G_t = 0
                    for r in reversed(rewards):
                        G_t = r + G_t
                        new_returns.append(G_t)
                    new_returns = list(reversed(new_returns))
                    returns.extend(new_returns)
                    episodes_sum_of_rewards.append(episode_cum_reward)
                    break
        # make loss
        returns = torch.tensor(returns)
        self.optimizer.zero_grad()
        log_probs = torch.cat(log_probs)
        loss = -(returns * log_probs).mean()
        print(", episode return ", '{:4.1f}'.format(np.mean(episodes_sum_of_rewards)),
              ", loss ", '{:4.1f}'.format(loss),
              sep='')
        # gradient step

        loss.backward()
        self.optimizer.step()
        return np.mean(episodes_sum_of_rewards)

    def train(self, env, nb_rollouts):
        avg_sum_rewards = []
        for ep in tqdm(range(nb_rollouts)):
            avg_sum_rewards.append(self.one_gradient_step(env))
        return avg_sum_rewards





class policyNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        state_dim = env.observation_space.shape[0]
        n_action = env.action_space.n
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, n_action)
        self.double()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(dim=0)
        x = F.relu(self.fc1(x))
        action_scores = self.fc2(x)
        return F.softmax(action_scores, dim=1)

    def sample_action(self, x):
        probabilities = self.forward(x)
        action_distribution = Categorical(probabilities)
        return action_distribution.sample().item()

    def log_prob(self, x, a):
        probabilities = self.forward(x)
        action_distribution = Categorical(probabilities)
        return action_distribution.log_prob(a)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {'learning_rate': 0.01,
              'nb_episodes': 1,
              'epsilon_min': 0.01,
              'epsilon_max': 1.,
              'epsilon_decay_period': 1000,
              'epsilon_delay_decay': 20,
              'gamma': 0.95
              }
    pi = policyNetwork(env)
    agent = ProjectAgent(config, pi)
    returns = agent.train(env, 50)
    plt.plot(returns)