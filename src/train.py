from sklearn.ensemble import RandomForestRegressor
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
from tqdm import tqdm
import pickle

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
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
    def __init__(self,):
        self.nb_actions = env.action_space.n
        self.Qfunction = None
    def act(self, observation, use_random=False):
        Qsa = []
        for a in range(self.nb_actions):
            sa = np.append(observation,a).reshape(1, -1)
            Qsa.append(self.Qfunction.predict(sa))
        return np.argmax(Qsa)
    
    def train(self, env, horizon=1000, iterations=1000, disable_tqdm=False, gamma=0.9):
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
    
    def save(self, path):
        payload = {"Qfunction": self.Qfunction, "nb_actions": self.nb_actions}
        pickle.dump(payload, open(path, "wb"))

    def load(self):
        payload = pickle.load(open("src/randomForestRegression.pkl", "rb"))
        self.Qfunction = payload["Qfunction"]
        self.nb_actions = payload["nb_actions"]
