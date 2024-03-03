from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
from tqdm import tqdm
import pickle
import torch
from torch import nn
import torch.nn.functional as F
import random

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

class QModel():
    """
    QModel class represents a model for Q-function approximation.

    Attributes:
        None

    Methods:
        __init__(): Initializes an instance of the QModel class.
        predict(sa): Predicts the Q-value for a given state-action pair.
        fit(sa, value): Fits the QModel to the given state-action pair and its corresponding value.

    """
    def __init__(self) -> None:
        pass
    def predict(self, sa):
        pass
    def fit(self, sa, value):
        pass


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        
        self.index = 0 # index of the next cell to be filled
        self.device = device
        self.data = []
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

def collect_samples(env, horizon=200, disable_tqdm=False, print_done_states=False):
    s, _ = env.reset()
    #dataset = []
    S = []
    A = []
    R = []
    S2 = []
    D = []
    for _ in tqdm(range(horizon), disable=disable_tqdm):
        a = env.action_space.sample()
        s2, r, done, trunc, _ = env.step(a)
        #dataset.append((s,a,r,s2,done,trunc))
        S.append(s)
        A.append(a)
        R.append(r)
        S2.append(s2)
        D.append(done)
        if done or trunc:
            s, _ = env.reset()
            if done and print_done_states:
                print("done!")
        else:
            s = s2
    S = np.array(S)
    A = np.array(A).reshape((-1,1))
    R = np.array(R)
    S2= np.array(S2)
    D = np.array(D)
    return S, A, R, S2, D
class ProjectAgent:
    def __init__(self):
        self.nb_actions = env.action_space.n
        self.Qfunction = None
    def act(self, observation, use_random=False):
        Qsa = []
        for a in range(self.nb_actions):
            sa = np.append(observation,a).reshape(1, -1)
            Qsa.append(self.Qfunction.predict(sa))
        return np.argmax(Qsa)
    
    def train(self, env, horizon=200, iterations=1000, disable_tqdm=False, gamma=0.9):
        # collecting samples
        S, A, R, S2, D = collect_samples(env, horizon, disable_tqdm=False, print_done_states=False)
        # training the agent
        nb_samples = S.shape[0]
        SA = np.append(S,A,axis=1)
        nb_actions = self.nb_actions
        for iter in tqdm(range(iterations), disable=disable_tqdm):
            if iter==0:
                value=R.copy()
            else:
                Q2 = np.zeros((nb_samples,nb_actions))
                for a2 in range(nb_actions):
                    A2 = a2*np.ones((S.shape[0],1))
                    S2A2 = np.append(S2,A2,axis=1)
                    Q2[:,a2] = self.Qfunction.predict(S2A2)
                max_Q2 = np.max(Q2,axis=1)
                value = R + gamma*(1-D)*max_Q2
            Q = RandomForestRegressor()
            Q.fit(SA,value)
            self.Qfunction = Q
        return self.Qfunction
    
    def train_dqn(self, env, horizon=1000, iterations=1000, disable_tqdm=False, gamma=0.9, verbose=True):
        # collecting samples
        S, A, R, S2, D = collect_samples(env, horizon, disable_tqdm=False, print_done_states=False)
        # training the agent
        nb_samples = S.shape[0]
        SA = np.append(S,A,axis=1)
        nb_actions = self.nb_actions
        input_dim = S.shape[1]+1
        output_dim = 1
        self.Qfunction = DQN(input_dim, output_dim)
        optimizer = torch.optim.Adam(self.Qfunction.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        for iter in tqdm(range(iterations), disable=disable_tqdm):
            if iter==0:
                value=R.copy()
            else:
                Q2 = np.zeros((nb_samples,nb_actions))
                for a2 in range(nb_actions):
                    A2 = a2*np.ones((S.shape[0],1))
                    S2A2 = np.append(S2,A2,axis=1)
                    Q2[:,a2] = self.Qfunction.predict(S2A2)
                max_Q2 = np.max(Q2,axis=1)
                value = R + gamma*(1-D)*max_Q2
            for i in range(nb_samples):
                sa = np.append(S[i],A[i]).reshape(1, -1)
                value_ = value[i]
                y = torch.tensor(value_, dtype=torch.float32)
                y_pred = self.Qfunction.forward(torch.tensor(sa, dtype=torch.float32))
                loss = criterion(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self.Qfunction
    
    def save(self, path):
        payload = {"Qfunction": self.Qfunction, "nb_actions": self.nb_actions}
        pickle.dump(payload, open(path, "wb"))

    def load(self):
        payload = pickle.load(open("src/randomForestRegression.pkl", "rb"))
        self.Qfunction = payload["Qfunction"]
        self.nb_actions = payload["nb_actions"]
def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()
class dqn_agent:
    def __init__(self, config, model):
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        print("device: ", device)
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
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(epsilon-self.epsilon_step, self.epsilon_min)
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
                # Monitoring
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
                          ", ep return ", '{:4.1f}'.format(episode_cum_reward), 
                          ", MC tot ", '{:6.2f}'.format(MC_tr),
                          ", MC disc ", '{:6.2f}'.format(MC_dr),
                          ", V0 ", '{:6.2f}'.format(V0),
                          sep='')
                else:
                    episode_return.append(episode_cum_reward)
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", '{:4.1f}'.format(episode_cum_reward), 
                          sep='')

                
                state, _ = env.reset()
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return, MC_avg_discounted_reward, MC_avg_total_reward, V_init_state
    def save(self, path):
        payload = {"model": self.model, "nb_actions": self.nb_actions, "target_model": self.target_model}
        pickle.dump(payload, open(path, "wb"))


state_dim = env.observation_space.shape[0]
# print(state_dim)
n_action = env.action_space.n 
# print(n_action)
nb_neurons=124
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
DQN = torch.nn.Sequential(nn.Linear(state_dim, nb_neurons),
                          nn.ReLU(),
                          nn.Linear(nb_neurons, nb_neurons),
                          nn.ReLU(), 
                          nn.Linear(nb_neurons, nb_neurons),
                          nn.ReLU(), 
                          nn.Linear(nb_neurons, n_action)).to(device)

# # DQN config
# config = {'nb_actions': n_action,
#           'learning_rate': 0.001,
#           'gamma': 0.95,
#           'buffer_size': int(1e5),
#           'epsilon_min': 0.01,
#           'epsilon_max': 1.,
#           'epsilon_decay_period': 200*15,
#           'epsilon_delay_decay': 200*5,
#           'batch_size': 40,
#           'gradient_steps': 1,
#           'update_target_strategy': 'replace', # or 'ema'
#           'update_target_freq': 50,
#           'update_target_tau': 0.005,
#           'criterion': torch.nn.SmoothL1Loss(),
#           'monitoring_nb_trials': 3}
# config = {'nb_actions': n_action,
#             'learning_rate': 0.001,
#             'gamma': 0.95,
#             'buffer_size': 1000000,
#             'epsilon_min': 0.01,
#             'epsilon_max': 1,
#             'epsilon_decay_period': 200*15,
#             'epsilon_delay_decay': 200*5,
#             'batch_size': 1024,
#             'gradient_steps': 10,
#             'update_target_strategy': 'ema', # or 'ema'
#             'update_target_freq': 50,
#             'update_target_tau': 0.0005,
#             'criterion': torch.nn.SmoothL1Loss(),
#             'monitoring_nb_trials': 3}

# # Train agent
# agent = dqn_agent(config, DQN)
# ep_length, disc_rewards, tot_rewards, V0 = agent.train(env, 200)
# agent.save("src/dqn.pkl")
# plt.plot(ep_length, label="training episode length")
# plt.plot(tot_rewards, label="MC eval of total reward")
# plt.legend()
# plt.figure()
# plt.plot(disc_rewards, label="MC eval of discounted reward")
# plt.plot(V0, label="average $max_a Q(s_0)$")
# plt.legend()
# agent = ProjectAgent()
# agent.train(env, horizon=int(1e4), iterations=300, disable_tqdm=False, gamma=0.9)
# agent.save("src/randomForestRegression.pkl")
# agent.train_dqn(env, horizon=1000, iterations=1000, disable_tqdm=False, gamma=0.9)
# agent.save("src/dqn.pkl")

class ProjectAgent:
    def __init__(self):
        self.nb_actions = env.action_space.n
        self.model = None
    def act(self, observation, use_random=False):
        return greedy_action(self.model, observation)
    
    def train(self, env, horizon=200, iterations=1000, disable_tqdm=False, gamma=0.9):
        # collecting samples
        S, A, R, S2, D = collect_samples(env, horizon, disable_tqdm=False, print_done_states=False)
        # training the agent
        nb_samples = S.shape[0]
        SA = np.append(S,A,axis=1)
        nb_actions = self.nb_actions
        for iter in tqdm(range(iterations), disable=disable_tqdm):
            if iter==0:
                value=R.copy()
            else:
                Q2 = np.zeros((nb_samples,nb_actions))
                for a2 in range(nb_actions):
                    A2 = a2*np.ones((S.shape[0],1))
                    S2A2 = np.append(S2,A2,axis=1)
                    Q2[:,a2] = self.Qfunction.predict(S2A2)
                max_Q2 = np.max(Q2,axis=1)
                value = R + gamma*(1-D)*max_Q2
            Q = RandomForestRegressor()
            Q.fit(SA,value)
            self.Qfunction = Q
        return self.Qfunction
    def load(self):
        model_dict = torch.load("src/dqn.pth", map_location=torch.device('cpu'))
        model = DQN
        model.load_state_dict(model_dict)
        self.model = model
        self.nb_actions = 4
        # payload = pickle.load(open("src/dqn.pkl", "rb"))
        # self.model = payload["model"]
        # self.nb_actions = payload["nb_actions"]

if __name__ == "__main__":
    # training
    # # DQN config
    config = {'nb_actions': n_action,
            'learning_rate': 0.001,
            'gamma': 0.95,
            'buffer_size': int(1e5),
            'epsilon_min': 0.01,
            'epsilon_max': 1.,
            'epsilon_decay_period': 200*15,
            'epsilon_delay_decay': 200*5,
            'batch_size': 40,
            'gradient_steps': 1,
            'update_target_strategy': 'replace', # or 'ema'
            'update_target_freq': 50,
            'update_target_tau': 0.005,
            'criterion': torch.nn.SmoothL1Loss(),
            'monitoring_nb_trials': 3}
    config = {'nb_actions': n_action,
                'learning_rate': 5e-4,
                'gamma': 0.95,
                'buffer_size': 2048*40,
                'epsilon_min': 0.05,
                'epsilon_max': 1,
                'epsilon_decay_period': 200*50,
                'epsilon_delay_decay': 200*10,
                'batch_size': 2048,
                'gradient_steps': 40,
                'update_target_strategy': 'ema', # or 'ema'
                'update_target_freq': 20,
                'update_target_tau': 0.005,
                'criterion': torch.nn.SmoothL1Loss(),
                'monitoring_nb_trials': 0}

    # Train agent
    agent = dqn_agent(config, DQN)
    ep_length, disc_rewards, tot_rewards, V0 = agent.train(env, 1000)
    agent.save("src/dqn.pkl")
    payload = pickle.load(open("src/dqn.pkl", "rb"))
    model = payload["target_model"]
    torch.save(model.state_dict(), "src/dqn.pth")
    