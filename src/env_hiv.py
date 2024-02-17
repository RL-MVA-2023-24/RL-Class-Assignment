import gymnasium as gym
import numpy as np


class HIVPatient(gym.Env):
    """HIV patient simulator

    Implements the simulator defined in 'Dynamic Multidrug Therapies for HIV: Optimal and STI Control Approaches' by Adams et al. (2004).
    The transition() function allows to simulate continuous time dynamics and control.
    The step() function is tailored for the evaluation of Structured Treatment Interruptions.
    """

    def __init__(
        self, clipping=True, logscale=False, domain_randomization: bool = False
    ):
        super(HIVPatient, self).__init__()

        self.domain_randomization = domain_randomization
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            shape=(6,), low=-np.inf, high=np.inf, dtype=np.float32
        )

        self.T1 = 163573.0  # healthy type 1 cells concentration (cells per mL)
        self.T1star = 11945.0  # infected type 1 cells concentration (cells per mL)
        self.T2 = 5.0  # healthy type 2 cells concentration (cells per mL)
        self.T2star = 46.0  # infected type 2 cells concentration (cells per mL)
        self.V = 63919.0  # free virus (copies per mL)
        self.E = 24.0  # immune effector cells concentration (cells per mL)

        # actions
        self.action_set = [
            np.array(pair) for pair in [[0.0, 0.0], [0.0, 0.3], [0.7, 0.0], [0.7, 0.3]]
        ]

        self._reset_patient_parameters()

        # action bounds
        self.a1 = 0.0  # lower bound on reverse transcriptase efficacy
        self.b1 = 0.7  # lower bound on reverse transcriptase efficacy
        self.a2 = 0.0  # lower bound on protease inhibitor efficacy
        self.b2 = 0.3  # lower bound on protease inhibitor efficacy

        # reward model
        self.Q = 0.1
        self.R1 = 20000.0
        self.R2 = 20000.0
        self.S = 1000.0

        # options
        self.clipping = clipping  # clip the state to the upper and lower bounds (affects the state attributes, which are clipped before being stored in T1, T1star, etc.)
        self.logscale = logscale  # convert state to log10-scale before returning (does not affect the state attributes, which remain stored without the log10 operation)
        self.T1Upper = 1e6
        self.T1starUpper = 5e4
        self.T2Upper = 3200.0
        self.T2starUpper = 80.0
        self.VUpper = 2.5e5
        self.EUpper = 353200.0
        self.upper = np.array(
            [
                self.T1Upper,
                self.T1starUpper,
                self.T2Upper,
                self.T2starUpper,
                self.VUpper,
                self.EUpper,
            ]
        )
        self.T1Lower = 0.0
        self.T1starLower = 0.0
        self.T2Lower = 0.0
        self.T2starLower = 0.0
        self.VLower = 0.0
        self.ELower = 0.0
        self.lower = np.array(
            [
                self.T1Lower,
                self.T1starLower,
                self.T2Lower,
                self.T2starLower,
                self.VLower,
                self.ELower,
            ]
        )

    def rawstate(self):
        return np.array([self.T1, self.T1star, self.T2, self.T2star, self.V, self.E])

    def state(self):
        s = np.array([self.T1, self.T1star, self.T2, self.T2star, self.V, self.E])
        if self.clipping:
            np.clip(s, self.lower, self.upper, out=s)
        if self.logscale:
            s = np.log10(s)
        return s

    def _reset_patient_parameters(self):
        if self.domain_randomization:
            # randomly changing patient parameters
            self.k1 = np.random.uniform(low=5e-7, high=8e-7)
            # cell2
            self.k2 = np.random.uniform(low=0.1e-4, high=1.0e-4)
            self.f = np.random.uniform(low=0.29, high=0.34)

        else:

            self.k1 = 8e-7  # infection rate (mL per virions and per day)

            self.k2 = 1e-4  # infection rate (mL per virions and per day)
            self.f = 0.34  # treatment efficacy reduction for type 2 cells
        # patient parameters

        # cell type 1
        self.lambda1 = 1e4  # production rate (cells per mL and per day)
        self.d1 = 1e-2  # death rate (per day)
        self.m1 = 1e-5  # immune-induced clearance rate (mL per cells and per day)
        self.rho1 = 1  # nb virions infecting a cell (virions per cell)

        # cell type 2
        self.lambda2 = 31.98  # production rate (cells per mL and per day)
        self.d2 = 1e-2  # death rate (per day)
        self.m2 = 1e-5  # immune-induced clearance rate (mL per cells and per day)
        self.rho2 = 1  # nb virions infecting a cell (virions per cell)
        # infected cells
        self.delta = 0.7  # death rate (per day)
        self.NT = 100  # virions produced (virions per cell)
        self.c = 13  # virus natural death rate (per day)
        # immune response (immune effector cells)
        self.lambdaE = 1  # production rate (cells per mL and per day)
        self.bE = 0.3  # maximum birth rate (per day)
        self.Kb = 100  # saturation constant for birth (cells per mL)
        self.dE = 0.25  # maximum death rate (per day)
        self.Kd = 500  # saturation constant for death (cells per mL)
        self.deltaE = 0.1  # natural death rate (per day)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None, mode="unhealthy"
    ):
        if mode == "uninfected":
            self.T1 = 1e6
            self.T1star = 0.0
            self.T2 = 3198.0
            self.T2star = 0.0
            self.V = 0.0
            self.E = 10.0
        elif mode == "unhealthy":
            self.T1 = 163573.0
            self.T1star = 11945.0
            self.T2 = 5.0
            self.T2star = 46.0
            self.V = 63919.0
            self.E = 24.0
        elif mode == "healthy":
            self.T1 = 967839.0
            self.T1star = 76.0
            self.T2 = 621.0
            self.T2star = 6.0
            self.V = 415.0
            self.E = 353108.0
        else:
            print("Patient mode '", mode, "' unrecognized. State unchanged.")

        self._reset_patient_parameters()

        return self.state(), {}

    def der(self, state, action):
        T1 = state[0]
        T1star = state[1]
        T2 = state[2]
        T2star = state[3]
        V = state[4]
        E = state[5]

        eps1 = action[0]
        eps2 = action[1]

        T1dot = self.lambda1 - self.d1 * T1 - self.k1 * (1 - eps1) * V * T1

        T1stardot = (
            self.k1 * (1 - eps1) * V * T1 - self.delta * T1star - self.m1 * E * T1star
        )
        T2dot = self.lambda2 - self.d2 * T2 - self.k2 * (1 - self.f * eps1) * V * T2
        T2stardot = (
            self.k2 * (1 - self.f * eps1) * V * T2
            - self.delta * T2star
            - self.m2 * E * T2star
        )
        Vdot = (
            self.NT * self.delta * (1 - eps2) * (T1star + T2star)
            - self.c * V
            - (
                self.rho1 * self.k1 * (1 - eps1) * T1
                + self.rho2 * self.k2 * (1 - self.f * eps1) * T2
            )
            * V
        )
        Edot = (
            self.lambdaE
            + self.bE * (T1star + T2star) * E / (T1star + T2star + self.Kb)
            - self.dE * (T1star + T2star) * E / (T1star + T2star + self.Kd)
            - self.deltaE * E
        )
        return np.array([T1dot, T1stardot, T2dot, T2stardot, Vdot, Edot])

    def transition(self, state, action, duration):
        """duration should be a multiple of 1e-3"""
        state0 = np.copy(state)
        state0_orig = np.copy(state)
        nb_steps = int(duration // 1e-3)
        for i in range(nb_steps):
            der = self.der(state0, action)
            state1 = state0 + der * 1e-3

            # np.clip(state1, self.lower, self.upper, out=state1)
            state0 = state1
        return state1

    def reward(self, state, action, state2):
        rew = -(
            self.Q * state[4]
            + self.R1 * action[0] ** 2
            + self.R2 * action[1] ** 2
            - self.S * state[5]
        )
        return rew

    def step(self, a_index):
        state = self.state()
        action = self.action_set[a_index]
        state2 = self.transition(state, action, 5)
        rew = self.reward(state, action, state2)
        if self.clipping:
            np.clip(state2, self.lower, self.upper, out=state2)

        self.T1 = state2[0]
        self.T1star = state2[1]
        self.T2 = state2[2]
        self.T2star = state2[3]
        self.V = state2[4]
        self.E = state2[5]

        if self.logscale:
            state2 = np.log10(state2)

        return state2, rew, False, False, {}
