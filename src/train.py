from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

import pickle
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
from joblib import dump, load

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

class ProjectAgent:
    def __init__(self, env):
      self.env = env
      self.iterations = 20
      self.gamma = 0.9
      self.Qfunc = None
      self.Qfunctions = []
      self.path = "bestmodel.joblib"
      self.S = []
      self.A = []
      self.R = []
      self.S2 = []
      self.D = []

    def collect_samples(self, env, horizon, epsilon = 0.0, disable_tqdm=False, print_done_states=False):
      s, _ = env.reset()
      #dataset = []
      for _ in range(horizon):
          if np.random.rand() < epsilon:
              a = self.greedy_action(self.Qfunc, s, self.env.action_space.n)
          else:
              a = self.env.action_space.sample()
          s2, r, done, trunc, _ = env.step(a)
          #dataset.append((s,a,r,s2,done,trunc))
          self.S.append(s)
          self.A.append(a)
          self.R.append(r)
          self.S2.append(s2)
          self.D.append(done)
          if done or trunc:
              s, _ = env.reset()
              if done and print_done_states:
                  print("done!")
          else:
              s = s2

    def rf_fqi(self, iterations, nb_actions, gamma, disable_tqdm=False):
      R = self.R
      S2 = self.S2
      D = np.array(self.D)
      nb_samples = len(self.S)
      S = np.array(self.S)
      A = np.array(self.A).reshape((-1, 1))
      SA = np.append(S,A,axis=1)
      Q = self.Qfunc
      for iter in range(iterations):
          if iter==0 and Q is None:
              value=R.copy()
          else:
              Q2 = np.zeros((nb_samples,nb_actions))
              for a2 in range(nb_actions):
                  A2 = a2*np.ones((S.shape[0],1))
                  S2A2 = np.append(S2,A2,axis=1)
                  Q2[:,a2] = Q.predict(S2A2)
              max_Q2 = np.max(Q2,axis=1)
              value = R + gamma*(1-D)*max_Q2
          Q = ExtraTreesRegressor(n_estimators = 70, max_depth= 5, n_jobs = -1)
          Q.fit(SA,value)
      self.Qfunc = Q

    def plot_res(self):
      s0, _ = self.env.reset()
      Vs0 = np.zeros(self.iterations)
      for i in range(self.iterations):
        Qs0a = []
        for a in range(self.env.action_space.n):
          s0a = np.append(s0,a).reshape(1, -1)
          Qs0a.append(self.Qfunctions[i].predict(s0a))
        Vs0[i] = np.max(Qs0a)
      plt.plot(Vs0)

    def greedy_action(self, Q,s,nb_actions):
      Qsa = []
      for a in range(nb_actions):
          sa = np.append(s,a).reshape(1, -1)
          Qsa.append(Q.predict(sa))
      return np.argmax(Qsa)

    def train(self, horizon, n_update=700, nb_iterations=50, nb_steps=25):
      self.collect_samples(self.env, horizon)
      self.rf_fqi(nb_iterations, self.env.action_space.n, self.gamma)

      eps_ini = 0.3
      eps_end = 0.98

      list_eps = np.linspace(eps_ini, eps_end, num = nb_steps)
      for step in range(nb_steps):
          print(step)
          self.collect_samples(self.env, n_update, epsilon= list_eps[step])
          self.rf_fqi(nb_iterations, self.env.action_space.n, self.gamma)

    def act(self, observation, use_random=False):
        if use_random:
          return np.randint(self.env.action_space.n)
        else:
          return self.greedy_action(self.Qfunc, observation, env.action_space.n)

    def savepkl(self, path= None):
      # Save the model to a file using pickle
      if path != None:
        self.path = path
      with open(self.path, 'wb') as file:
        pickle.dump(self.Qfunc, file)

    def loadpkl(self):
      # Load the model back using pickle
      with open(self.path, 'rb') as file:
        print(pickle.load(file))
        self.Qfunc = pickle.load(file)

    def save(self, path = None):
        if path != None:
          self.path=path
        dump(self.Qfunc, self.path)

    def load(self):
        self.Qfunc = load('bestmodel.joblib')
