import os

import imageio
import numpy as np
import torch
from gomoku_mainer import Opponent
from custom_environment.gomoku.env import gomoku
from PIL import Image, ImageDraw, ImageFont

from agilerl.algorithms.dqn import DQN

# Define function to return image
def _label_with_episode_number(frame, episode_num, frame_no, p):
    im = Image.fromarray(frame)
    drawer = ImageDraw.Draw(im)
    text_color = (255, 255, 255)
    font = ImageFont.truetype("DejaVuSerif.ttf", size=25)
    drawer.text(
        (100, 40),
        f"Episode: {episode_num+1}     Frame: {frame_no}",
        fill=text_color,
        font=font,
    )
    if p == 1:
        player = "Player 1"
        color = (255, 0, 0)
    if p == 2:
        player = "Player 2"
        color = (100, 255, 150)
    if p is None:
        player = "Self-play"
        color = (255, 255, 255)
    drawer.text((700, 40), f"Agent: {player}", fill=color, font=font)
    return im


# Resizes frames to make file size smaller
def resize_frames(frames, fraction):
    resized_frames = []
    for img in frames:
        new_width = int(img.width * fraction)
        new_height = int(img.height * fraction)
        img_resized = img.resize((new_width, new_height))
        resized_frames.append(np.array(img_resized))

    return resized_frames

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = "./models/DQN-gomoku2/lesson4_trained_agent.pt"
    path1 = "./models/DQN-gomoku2/lesson1_trained_agent.pt"  # Path to saved agent checkpoint
    path2 = "./models/DQN-gomoku2/lesson2_trained_agent.pt"
    path3 = "./models/DQN-gomoku2/lesson3_trained_agent.pt"
    path4 = "./models/DQN-gomoku2/lesson4_trained_agent.pt"

    env = gomoku.env(render_mode="rgb_array")
    env.reset()

    # Configure the algo input arguments
    state_dim = [
        env.observation_space(agent)["observation"].shape for agent in env.agents
    ]
    one_hot = False
    action_dim = [env.action_space(agent).n for agent in env.agents]

    # Pre-process dimensions for pytorch layers
    # We will use self-play, so we only need to worry about the state dim of a single agent
    # We flatten the 6x7x2 observation as input to the agent's neural network
    state_dim = np.moveaxis(np.zeros(state_dim[0]), [-1], [-3]).shape
    action_dim = action_dim[0]

    # Instantiate an DQN object
    dqn = DQN(
        state_dim,
        action_dim,
        one_hot,
        device=device,
    )

    dqn1 = DQN(
        state_dim,
        action_dim,
        one_hot,
        device=device,
    )

    dqn2 = DQN(
        state_dim,
        action_dim,
        one_hot,
        device=device,
    )

    dqn3 = DQN(
        state_dim,
        action_dim,
        one_hot,
        device=device,
    )

    dqn4 = DQN(
        state_dim,
        action_dim,
        one_hot,
        device=device,
    )

    # Load the saved algorithm into the DQN object
    dqn.loadCheckpoint(path)
    dqn1.loadCheckpoint(path1)
    dqn2.loadCheckpoint(path2)
    dqn3.loadCheckpoint(path3)
    dqn4.loadCheckpoint(path4)

    for opponent_difficulty in ["lesson1", "lesson2", "lesson3", "lesson4"]: #["random", "weak", "strong", "self"]:
        # Create opponent
        if opponent_difficulty == "lesson1":
            opponent = dqn1
        elif opponent_difficulty == "lesson2":
            opponent = dqn2
        elif opponent_difficulty == "lesson3":
            opponent = dqn3
        else:
            opponent = dqn4

        # Define test loop parameters
        episodes = 1  # Number of episodes to test agent on
        max_steps = (
            500  # Max number of steps to take in the environment in each episode
        )

        rewards = []  # List to collect total episodic reward
        frames = []  # List to collect frames

        print("============================================")
        print(f"Agent: {path}")
        print(f"Opponent: {opponent_difficulty}")

        # Test loop for inference
        for ep in range(episodes):
            opponent_first = False
            p = None

            env.reset()  # Reset environment at start of episode
            frame = env.render()
            frames.append(
                _label_with_episode_number(frame, episode_num=ep, frame_no=0, p=p)
            )
            observation, reward, done, truncation, _ = env.last()
            player = -1  # Tracker for which player's turn it is
            score = 0
            for idx_step in range(max_steps):
                #action_mask8 = np.array([int(i) for i in observation['action_mask']])
                action_mask8 = observation["action_mask"]
                if player < 0:
                    state = np.moveaxis(observation["observation"], [-1], [-3])
                    state = np.expand_dims(state, 0)
                    if opponent_first:
                        action = opponent.getAction(
                            state, epsilon=0, action_mask=action_mask8
                        )[0]
                    else:
                        action = dqn.getAction(
                            state, epsilon=0, action_mask=action_mask8
                        )[
                            0
                        ]  # Get next action from agent
                if player > 0:
                    state = np.moveaxis(observation["observation"], [-1], [-3])
                    state[[0, 1], :, :] = state[[0, 1], :, :]
                    state = np.expand_dims(state, 0)
                    if not opponent_first:
                        action = opponent.getAction(
                            state, epsilon=0, action_mask=action_mask8
                        )[0]
                    else:
                        action = dqn.getAction(
                            state, epsilon=0, action_mask=action_mask8
                        )[
                            0
                        ]  # Get next action from agent
                
                env.step(action)  # Act in environment
                observation, reward, termination, truncation, _ = env.last()
                frame = env.render()
                frames.append(
                    _label_with_episode_number(
                        frame, episode_num=ep, frame_no=idx_step, p=p
                    )
                )

                if truncation or termination:
                    env.step(None)
                    observation, reward, termination, truncation, _ = env.last()
                    
                if (player > 0 and opponent_first) or (
                    player < 0 and not opponent_first
                ):
                    score += reward
                else:
                    score -= reward

                # Stop episode if any agents have terminated
                if truncation or termination:
                    break

                player *= -1

            print("-" * 15, f"Episode: {ep+1}", "-" * 15)
            print(f"Episode length: {idx_step}")
            print(f"Score: {score}")

        print("============================================")
        frames = resize_frames(frames, 0.5)

        # Save the gif to specified path
        gif_path = "./videos/"
        os.makedirs(gif_path, exist_ok=True)
        imageio.mimwrite(
            os.path.join("./videos/", f"gomoku_lesson4_vs_{opponent_difficulty}.gif"),
            frames,
            duration=400,
            loop=True,
        )
    env.close()