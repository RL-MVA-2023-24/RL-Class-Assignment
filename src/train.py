from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.
state_dim = env.observation_space.shape[0]
n_action = env.action_space.n
# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
import numpy as np

class policyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128, dtype = torch.float64)
        self.fc2 = nn.Linear(128, n_action, dtype = torch.float64)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(dim=0)
        x = F.relu(self.fc1(x))
        action_scores = self.fc2(x)
        return F.softmax(action_scores,dim=1)

    def sample_action(self, x):
        probabilities = self.forward(x)
        action_distribution = Categorical(probabilities)
        return action_distribution.sample().item()

    def log_prob(self, x, a):
        probabilities = self.forward(x)
        action_distribution = Categorical(probabilities)
        return action_distribution.log_prob(a)


class valueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128, dtype = torch.float64)
        self.fc2 = nn.Linear(128, 1, dtype = torch.float64)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(dim=0)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class a2c_agent:
    def __init__(self, config, policy_network, value_network):
        self.device = "cuda" if next(policy_network.parameters()).is_cuda else "cpu"
        self.scalar_dtype = next(policy_network.parameters()).dtype
        self.policy = policy_network
        self.value = value_network
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.99
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = torch.optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()),lr=lr)
        self.nb_episodes = config['nb_episodes'] if 'nb_episodes' in config.keys() else 1
        self.entropy_coefficient = config['entropy_coefficient'] if 'entropy_coefficient' in config.keys() else 0.001

    def sample_action(self, x):
        probabilities = self.policy(torch.as_tensor(x))
        action_distribution = Categorical(probabilities)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)
        entropy = action_distribution.entropy()
        return action.item(), log_prob, entropy
    
    def one_gradient_step(self, env):
        # run trajectories until done
        episodes_sum_of_rewards = []
        log_probs = []
        returns = []
        values = []
        entropies = []
        for ep in range(self.nb_episodes):
            x,_ = env.reset()
            rewards = []
            episode_cum_reward = 0
            while(True):
                a, log_prob, entropy = self.sample_action(x)
                y,r,d,t,_ = env.step(a)
                values.append(self.value(torch.as_tensor(x)).squeeze(dim=0))
                log_probs.append(log_prob)
                entropies.append(entropy)
                rewards.append(r)
                episode_cum_reward += r
                x=y
                if d or t:
                    # compute returns-to-go
                    new_returns = []
                    G_t = self.value(torch.as_tensor(x)).squeeze(dim=0)
                    for r in reversed(rewards):
                        G_t = r + self.gamma * G_t
                        new_returns.append(G_t)
                    new_returns = list(reversed(new_returns))
                    returns.extend(new_returns)
                    episodes_sum_of_rewards.append(episode_cum_reward)
                    break
        # make loss        
        returns = torch.cat(returns)
        values = torch.cat(values)
        log_probs = torch.cat(log_probs)
        entropies = torch.cat(entropies)
        advantages = returns - values
        pg_loss = -(advantages.detach() * log_probs).mean()
        entropy_loss = -entropies.mean()
        critic_loss = advantages.pow(2).mean()
        loss = pg_loss + critic_loss + self.entropy_coefficient * entropy_loss
        # gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return np.mean(episodes_sum_of_rewards)

    def train(self, env, nb_rollouts):
        avg_sum_rewards = []
        for ep in range(nb_rollouts):
            avg_sum_rewards.append(self.one_gradient_step(env))
        return avg_sum_rewards


    
class ProjectAgent:

    def __init__(self):
        config = {'gamma':0.99,
          'learning_rate':0.1,
          'nb_episodes':10,
          'entropy_coefficient': 1e-3}
        
        self.policy = policyNetwork()
        self.value = valueNetwork()
        self.device = "cuda" if next(self.policy.parameters()).is_cuda else "cpu"
        self.scalar_dtype = next(self.policy.parameters()).dtype

        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.99
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = torch.optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()),lr=lr)
        self.nb_episodes = config['nb_episodes'] if 'nb_episodes' in config.keys() else 1
        self.entropy_coefficient = config['entropy_coefficient'] if 'entropy_coefficient' in config.keys() else 0.001

    def sample_action(self, x):
        probabilities = self.policy(torch.as_tensor(x))
        action_distribution = Categorical(probabilities)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)
        entropy = action_distribution.entropy()
        return action.item(), log_prob, entropy
    
    def one_gradient_step(self, env):
        # run trajectories until done
        episodes_sum_of_rewards = []
        log_probs = []
        returns = []
        values = []
        entropies = []
        for ep in range(self.nb_episodes):
            x,_ = env.reset()
            rewards = []
            episode_cum_reward = 0
            while(True):
                a, log_prob, entropy = self.sample_action(x)
                # print( f'action = {a}')
                y,r,d,t,_ = env.step(a)
                values.append(self.value(torch.as_tensor(x)).squeeze(dim=0))
                log_probs.append(log_prob)
                entropies.append(entropy)
                rewards.append(r)
                episode_cum_reward += r
                x=y
                if d or t:
                    # compute returns-to-go
                    new_returns = []
                    G_t = self.value(torch.as_tensor(x)).squeeze(dim=0)
                    for r in reversed(rewards):
                        G_t = r + self.gamma * G_t
                        new_returns.append(G_t)
                    new_returns = list(reversed(new_returns))
                    returns.extend(new_returns)
                    episodes_sum_of_rewards.append(episode_cum_reward)
                    break
        # make loss        
        returns = torch.cat(returns)
        values = torch.cat(values)
        log_probs = torch.cat(log_probs)
        entropies = torch.cat(entropies)
        advantages = returns - values
        pg_loss = -(advantages.detach() * log_probs).mean()
        entropy_loss = -entropies.mean()
        critic_loss = advantages.pow(2).mean()
        loss = pg_loss + critic_loss + self.entropy_coefficient * entropy_loss
        # gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        mean_sum_of_rewards = np.mean(episodes_sum_of_rewards)
        print('rewards:',mean_sum_of_rewards)
        return mean_sum_of_rewards

    def train(self, env, nb_rollouts):
        avg_sum_rewards = []
        for ep in range(nb_rollouts):
            avg_sum_rewards.append(self.one_gradient_step(env))
        return avg_sum_rewards
    
    def act(self, observation, use_random=False):
        if use_random:
            x,_,_ = self.sample_action(observation)
            # print(f'obsvervation:{observation}')
            # print(f'act:{x}')
            return x

        probabilities = self.policy(torch.as_tensor(observation))
        return probabilities.argmax().item()

    def save(self, path):
        torch.save(self.policy.state_dict(), 'policy_'+ path)
        torch.save(self.value.state_dict(),'value_'+ path)
        print(f'Model saved under {path}')
        # pass
        # torch.save(self.policy.state_dict(), path + '/policy_weights.pth')
        # torch.save(self.value.state_dict(),  path +'/value_weights.pth')
        # print(f'Model saved at {path}')

    def load(self):
        self.policy.load_state_dict(torch.load('src/policy_weights.pth', map_location=torch.device('cpu')))
        self.policy.eval()
        self.value.load_state_dict(torch.load('src/value_weights.pth',  map_location=torch.device('cpu')))
        self.value.eval()
        
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)
            
def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()

from copy import deepcopy

class ProjectAgent:

    def __init__(self):
        nb_neurons = 512
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.nn.Sequential(nn.Linear(state_dim, nb_neurons),
                          nn.ReLU(),
                          nn.Linear(nb_neurons, nb_neurons),
                          nn.ReLU(), 
                          nn.Linear(nb_neurons, nb_neurons),
                          nn.ReLU(), 
                          nn.Linear(nb_neurons, n_action)).to(device)
        config = {'nb_actions': env.action_space.n,
          'learning_rate': 1e-2,
          'gamma': 0.999,
          'buffer_size': 100_000,
          'epsilon_min': 0.01,
          'epsilon_max': 1.,
          'epsilon_decay_period': 20_000,
          'epsilon_delay_decay': 500,
          'batch_size': 200,
          'gradient_steps': 2,
          'update_target_strategy': 'replace', # or 'ema'
          'update_target_freq': 500,
          'update_target_tau': 0.005,
          'criterion': torch.nn.SmoothL1Loss()}
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size,device)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = model 
        self.target_model = deepcopy(self.model).to(device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        best_rew = 0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                if (episode + 1) % 100 == 0:
                  torch.save(self.model.state_dict(), f'true_dqn_weights_{episode + 1}.pth')
                if episode_cum_reward > best_rew:
                  best_rew = episode_cum_reward
                  torch.save(self.model.state_dict(), f'best_dqn_weights_{episode + 1}.pth')

                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:1.1e}'.format(episode_cum_reward),
                      sep='')
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return
    
    def act(self, observation, use_random=False):
        if use_random:
            action = env.action_space.sample()
        else:
            action = greedy_action(self.model, observation)
        return action

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        self.model.load_state_dict(torch.load('src/best_dqn_weights_169.pth', map_location=torch.device('cpu')))
        self.target_model = deepcopy(self.model).to(device)

# pi = policyNetwork(env).to(device)
# V  = valueNetwork(env).to(device)
# # agent = a2c_agent(config, pi, V)
# agent = ProjectAgent(config,pi,V)
# agent.load()

# from evaluate import evaluate_HIV, evaluate_HIV_population
# # Keep the following lines to evaluate your agent unchanged.
# score_agent: float = evaluate_HIV(agent=agent, nb_episode=1)
# score_agent_dr: float = evaluate_HIV_population(agent=agent, nb_episode=15)
# with open(file="score.txt", mode="w") as f:
#     f.write(f"{score_agent}\n{score_agent_dr}")
# agent = ProjectAgent()
# agent.train(env,50)
# agent.save('weight2.pth')

# from evaluate import evaluate_HIV, evaluate_HIV_population
# import random
# import os


# def seed_everything(seed: int = 42):
#     random.seed(seed)
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     torch.cuda.manual_seed_all(seed)


# # Initialization of the agent. Replace DummyAgent with your custom agent implementation.
# seed_everything(seed=42)
# # Keep the following lines to evaluate your agent unchanged.
# score_agent: float = evaluate_HIV(agent=agent, nb_episode=1)
# score_agent_dr: float = evaluate_HIV_population(agent=agent, nb_episode=15)
# with open(file="score2.txt", mode="w") as f:
#     f.write(f"{score_agent}\n{score_agent_dr}")