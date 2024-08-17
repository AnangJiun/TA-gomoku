from custom_environment.gomoku.env import gomoku
import numpy as np


env = gomoku.env()
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        env.step(None)
        observation, reward, termination, truncation, info = env.last()
        action = None
        break
    else:
        mask = observation["action_mask"]
        # this is where you would insert your policy
        action = env.action_space(agent).sample(mask)
    
    
    env.step(action)
    print(env.agent_selection)
    print(env.rewards, reward, termination, action)

print(reward, env.agent_selection, agent, env.rewards)
