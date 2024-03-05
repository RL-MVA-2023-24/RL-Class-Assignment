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

    
state_dim = state_dim = env.observation_space.shape[0]
n_action = env.action_space.n

class ProjectAgent:
    def __init__(self):
        nb_neurons= 512
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
        self.monitoring_nb_trials = config['monitoring_nb_trials'] if 'monitoring_nb_trials' in config.keys() else 0

    def MC_eval(self, env, nb_trials):   # NEW NEW NEW
        MC_total_reward = []
        MC_discounted_reward = []
        for _ in range(nb_trials):
            x,_ = env.reset()
            done = False
            trunc = False
            total_reward = 0
            discounted_reward = 0
            step = 0
            while not (done or trunc):
                a = greedy_action(self.model, x)
                y,r,done,trunc,_ = env.step(a)
                x = y
                total_reward += r
                discounted_reward += self.gamma**step * r
                step += 1
            MC_total_reward.append(total_reward)
            MC_discounted_reward.append(discounted_reward)
        return np.mean(MC_discounted_reward), np.mean(MC_total_reward)
    
    def V_initial_state(self, env, nb_trials):   # NEW NEW NEW
        with torch.no_grad():
            for _ in range(nb_trials):
                val = []
                x,_ = env.reset()
                val.append(self.model(torch.Tensor(x).unsqueeze(0).to(device)).max().item())
        return np.mean(val)
    
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
        MC_avg_total_reward = []   # NEW NEW NEW
        MC_avg_discounted_reward = []   # NEW NEW NEW
        V_init_state = []   # NEW NEW NEW
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
                    target_state_dict[key] = tau*model_state_dict + (1-tau)*target_state_dict
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                # Monitoring
                if (episode + 1) % 100 == 0:
                  torch.save(self.model.state_dict(), f'true_new_dqn_weights_{episode + 1}.pth')
                  torch.save(self.target_model.state_dict(), f'true_new_target_dqn_weights_{episode + 1}.pth')
                if episode_cum_reward > 1e9 and episode_cum_reward > best_rew:
                  best_rew = episode_cum_reward
                  torch.save(self.model.state_dict(), f'best_new_dqn_weights_{episode + 1}.pth')
                  torch.save(self.target_model.state_dict(), f'best_new_target_dqn_weights_{episode + 1}.pth')
                if self.monitoring_nb_trials>0:
                    MC_dr, MC_tr = self.MC_eval(env, self.monitoring_nb_trials)    # NEW NEW NEW
                    V0 = self.V_initial_state(env, self.monitoring_nb_trials)   # NEW NEW NEW
                    MC_avg_total_reward.append(MC_tr)   # NEW NEW NEW
                    MC_avg_discounted_reward.append(MC_dr)   # NEW NEW NEW
                    V_init_state.append(V0)   # NEW NEW NEW
                    episode_return.append(episode_cum_reward)   # NEW NEW NEW
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", '{:1.2e}'.format(episode_cum_reward), 
                          ", MC tot ", '{:6.2f}'.format(MC_tr),
                          ", MC disc ", '{:6.2f}'.format(MC_dr),
                          ", V0 ", '{:6.2f}'.format(V0),
                          sep='')
                else:
                    episode_return.append(episode_cum_reward)
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", '{:1.2e}'.format(episode_cum_reward), 
                          sep='')

                
                state, _ = env.reset()
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return, MC_avg_discounted_reward, MC_avg_total_reward, V_init_state
        
    def act(self, observation, use_random=False):
        if use_random:
            action = env.action_space.sample()
        else:
            action = greedy_action(self.model, observation)
        return action

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        self.model.load_state_dict(torch.load('src/Z_dqn_weights_175.pth', map_location=torch.device('cpu')))
        self.target_model = deepcopy(self.model).to(device)


if __name__ == "__main__":
    agent = ProjectAgent()
    agent.train(env,300)
    agent.save('saved_weights.pth')