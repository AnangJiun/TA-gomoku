from custom_environment.gomoku.env import gomoku
import numpy as np


env = gomoku.env()
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        mask = observation["action_mask"]
        # this is where you would insert your policy
        action = env.action_space(agent).sample(mask)

    env.step(action)

print(list(mask))
print(np.array([1 for i in range(225)]))
