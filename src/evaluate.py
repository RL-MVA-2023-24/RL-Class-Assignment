from statistics import mean
from functools import partial
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from env_hiv import HIVPatient
from interface import Agent


def evaluate_agent(agent: Agent, env: gym.Env, nb_episode: int = 10) -> float:
    """
    Evaluate an agent in a given environment.

    Args:
        agent (Agent): The agent to evaluate.
        env (gym.Env): The environment to evaluate the agent in.
        nb_episode (int): The number of episode to evaluate the agent.

    Returns:
        float: The mean reward of the agent over the episodes.
    """
    rewards: list[float] = []
    for _ in range(nb_episode):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        while not done and not truncated:
            action = agent.act(obs)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    return mean(rewards)


evaluate_HIV = partial(
    evaluate_agent, env=TimeLimit(HIVPatient(), max_episode_steps=200)
)


evaluate_HIV_population = partial(
    evaluate_agent,
    env=TimeLimit(HIVPatient(domain_randomization=True), max_episode_steps=200),
)
