"""This tutorial shows how to train a DQN agent on the connect four environment, using curriculum learning and self play.

Author: Nick (https://github.com/nicku-a)
"""
import copy
import os
import random
from collections import deque
from datetime import datetime

import numpy as np
import torch
import wandb
import yaml
from custom_environment.gomoku.env import gomoku
from tqdm import tqdm, trange

from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import initialPopulation
from agilerl.algorithms.dqn import DQN

class CurriculumEnv:
    """Wrapper around environment to modify reward for curriculum learning.

    :param env: Environment to learn in
    :type env: PettingZoo-style environment
    :param lesson: Lesson settings for curriculum learning
    :type lesson: dict
    """

    def __init__(self, env, lesson):
        self.env = env
        self.lesson = lesson
        self.boardsize = 15**2 - 1
        self.length = 5

    def fill_replay_buffer(self, memory, opponent):
        """Fill the replay buffer with experiences collected by taking random actions in the environment.

        :param memory: Experience replay buffer
        :type memory: AgileRL experience replay buffer
        """
        print("Filling replay buffer ...")

        pbar = tqdm(total=memory.memory_size)
        while len(memory) < memory.memory_size:
            # Randomly decide whether random player will go first or second
            if random.random() > 0.5:
                opponent_first = False
            else:
                opponent_first = True

            mem_full = len(memory)
            self.reset()  # Reset environment at start of episode
            observation, reward, done, truncation, _ = self.last()

            (
                p1_state,
                p1_state_flipped,
                p1_action,
                p1_next_state,
                p1_next_state_flipped,
            ) = (None, None, None, None, None)
            done, truncation = False, False

            while not (done or truncation):
                # Player 0's turn
                #p0_action_mask64 = np.array([int(i) for i in observation['action_mask']])
                p0_action_mask8 = observation["action_mask"]
                p0_state = np.moveaxis(observation["observation"], [-1], [-3])
                p0_state_flipped = np.expand_dims(np.flip(p0_state, 2), 0)
                p0_state = np.expand_dims(p0_state, 0)
                if opponent_first:
                    p0_action = self.env.action_space("player_0").sample(p0_action_mask8)
                else:
                    if self.lesson["warm_up_opponent"] == "random":
                        p0_action = opponent.getAction(
                            p0_action_mask8, p1_action, self.lesson["block_vert_coef"]
                        )
                    else:
                        p0_action = opponent.getAction(player=0)
                self.step(p0_action)  # Act in environment
                observation, env_reward, done, truncation, _ = self.last()
                p0_next_state = np.moveaxis(observation["observation"], [-1], [-3])
                p0_next_state_flipped = np.expand_dims(np.flip(p0_next_state, 2), 0)
                p0_next_state = np.expand_dims(p0_next_state, 0)

                if done or truncation:
                    reward = self.reward(done=True, player=0)
                    memory.save2memoryVectEnvs(
                        np.concatenate(
                            (p0_state, p1_state, p0_state_flipped, p1_state_flipped)
                        ),
                        [p0_action, p1_action, self.boardsize - p0_action, self.boardsize - p1_action], #gimana flip
                        [
                            reward,
                            self.lesson["rewards"]["lose"],
                            reward,
                            self.lesson["rewards"]["lose"],
                        ],
                        np.concatenate(
                            (
                                p0_next_state,
                                p1_next_state,
                                p0_next_state_flipped,
                                p1_next_state_flipped,
                            )
                        ),
                        [done, done, done, done],
                    )
                else:  # Play continues
                    if p1_state is not None:
                        reward = self.reward(done=False, player=1)
                        memory.save2memoryVectEnvs(
                            np.concatenate((p1_state, p1_state_flipped)),
                            [p1_action, self.boardsize - p1_action],
                            [reward, reward],
                            np.concatenate((p1_next_state, p1_next_state_flipped)),
                            [done, done],
                        )

                    # Player 1's turn
                    #p1_action_mask64 = np.array([int(i) for i in observation['action_mask']])
                    p1_action_mask8 = observation["action_mask"]
                    p1_state = np.moveaxis(observation["observation"], [-1], [-3])
                    p1_state[[0, 1], :, :] = p1_state[[0, 1], :, :]
                    p1_state_flipped = np.expand_dims(np.flip(p1_state, 2), 0)
                    p1_state = np.expand_dims(p1_state, 0)
                    if not opponent_first:
                        p1_action = self.env.action_space("player_1").sample(
                            p1_action_mask8
                        )
                    else:
                        if self.lesson["warm_up_opponent"] == "random":
                            p1_action = opponent.getAction(
                                p1_action_mask8, p0_action, self.lesson["block_vert_coef"]
                            )
                        else:
                            p1_action = opponent.getAction(player=1)
                    self.step(p1_action)  # Act in environment
                    observation, env_reward, done, truncation, _ = self.last()
                    p1_next_state = np.moveaxis(observation["observation"], [-1], [-3])
                    p1_next_state[[0, 1], :, :] = p1_next_state[[0, 1], :, :]
                    p1_next_state_flipped = np.expand_dims(np.flip(p1_next_state, 2), 0)
                    p1_next_state = np.expand_dims(p1_next_state, 0)

                    if done or truncation:
                        reward = self.reward(done=True, player=1)
                        memory.save2memoryVectEnvs(
                            np.concatenate(
                                (p0_state, p1_state, p0_state_flipped, p1_state_flipped)
                            ),
                            [p0_action, p1_action, self.boardsize - p0_action, self.boardsize - p1_action],
                            [
                                self.lesson["rewards"]["lose"],
                                reward,
                                self.lesson["rewards"]["lose"],
                                reward,
                            ],
                            np.concatenate(
                                (
                                    p0_next_state,
                                    p1_next_state,
                                    p0_next_state_flipped,
                                    p1_next_state_flipped,
                                )
                            ),
                            [done, done, done, done],
                        )

                    else:  # Play continues
                        reward = self.reward(done=False, player=0)
                        memory.save2memoryVectEnvs(
                            np.concatenate((p0_state, p0_state_flipped)),
                            [p0_action, self.boardsize - p0_action],
                            [reward, reward],
                            np.concatenate((p0_next_state, p0_next_state_flipped)),
                            [done, done],
                        )

            pbar.update(len(memory) - mem_full)
        pbar.close()
        print("Replay buffer warmed up.")
        return memory

    def check_winnable(self, lst, piece):
        """Checks if four pieces in a row represent a winnable opportunity, e.g. [1, 1, 1, 0] or [2, 0, 2, 2].

        :param lst: List of pieces in row
        :type lst: List
        :param piece: Player piece we are checking (1 or 2)
        :type piece: int
        """
        return lst.count(piece) == (self.length-1) and lst.count(0) == 1

    def check_vertical_win(self, player):
        """Checks if a win is vertical.

        :param player: Player who we are checking, 0 or 1
        :type player: int
        """
        board = np.array(self.env.env.board.squares).reshape(15,15)
        piece = player + 1

        column_count = 15
        row_count = 15

        # Check vertical locations for win
        for c in range(column_count):
            for r in range(row_count - (self.length - 1)):
                if (
                    board[r][c] == piece
                    and board[r + 1][c] == piece
                    and board[r + 2][c] == piece
                    and board[r + 3][c] == piece
                    and board[r + 4][c] == piece
                ):
                    return True
        return False

    def check_two_in_row(self, player):
        """Checks if there are three pieces in a row and a blank space next, or two pieces - blank - piece.

        :param player: Player who we are checking, 0 or 1
        :type player: int
        """
        board = np.array(self.env.env.board.squares).reshape(15,15)
        piece = player + 1

        # Check horizontal locations
        column_count = 15
        row_count = 15
        two_in_row_count = 0

        # Check vertical locations
        for c in range(column_count):
            for r in range(row_count - (self.length - 1)):
                if self.check_winnable(board[r : r + self.length, c].tolist(), piece):
                    two_in_row_count += 1

        # Check horizontal locations
        for r in range(row_count):
            for c in range(column_count - (self.length - 1)):
                if self.check_winnable(board[r, c : c + self.length].tolist(), piece):
                    two_in_row_count += 1

        # Check positively sloped diagonals
        for c in range(column_count - (self.length - 1)):
            for r in range(row_count - (self.length - 1)):
                if self.check_winnable(
                    [
                        board[r, c],
                        board[r + 1, c + 1],
                        board[r + 2, c + 2],
                        board[r + 3, c + 3],
                        board[r + 4, c + 4],
                    ],
                    piece,
                ):
                    two_in_row_count += 1

        # Check negatively sloped diagonals
        for c in range(column_count - (self.length-1)):
            for r in range((self.length-1), row_count):
                if self.check_winnable(
                    [
                        board[r, c],
                        board[r - 1, c + 1],
                        board[r - 2, c + 2],
                        board[r - 3, c + 3],
                        board[r - 4, c + 4],
                    ],
                    piece,
                ):
                    two_in_row_count += 1

        return two_in_row_count

    def reward(self, done, player):
        """Processes and returns reward from environment according to lesson criteria.

        :param done: Environment has terminated
        :type done: bool
        :param player: Player who we are checking, 0 or 1
        :type player: int
        """
        if done:
            reward = (
                self.lesson["rewards"]["vertical_win"]
                if self.check_vertical_win(player)
                else self.lesson["rewards"]["win"]
            )
        else:
            agent_two_count = self.check_two_in_row(1 - player)
            opp_two_count = self.check_two_in_row(player)
            if (agent_two_count + opp_two_count) == 0:
                reward = self.lesson["rewards"]["play_continues"]
            else:
                reward = (
                    self.lesson["rewards"]["three_in_row"] * agent_two_count
                    + self.lesson["rewards"]["opp_three_in_row"] * opp_two_count
                )
        return reward

    def last(self):
        """Wrapper around PettingZoo env last method."""
        return self.env.last()

    def step(self, action):
        """Wrapper around PettingZoo env step method."""
        self.env.step(action)

    def reset(self):
        """Wrapper around PettingZoo env reset method."""
        self.env.reset()

class Opponent:
    def __init__(self, env, difficulty):
        self.env = env.env
        self.difficulty = difficulty
        if self.difficulty == "random":
            self.getAction = self.random_opponent
        elif self.difficulty == "weak":
            self.getAction = self.weak_rule_based_opponent
        else:
            self.getAction = self.strong_rule_based_opponent
        self.num_rows = 15
        self.num_cols = 15
        self.length = 5  # For Gomoku, the winning length is 5 in a row
        self.boardsize = self.num_rows * self.num_cols

    def random_opponent(self, action_mask, last_opp_move=None, block_vert_coef=1):
        """Takes move for random opponent. If the lesson aims to randomly block vertical wins with a higher probability, this is done here too.

        :param action_mask: Mask of legal actions: 1=legal, 0=illegal
        :type action_mask: List
        :param last_opp_move: Most recent action taken by agent against this opponent
        :type last_opp_move: int
        :param block_vert_coef: How many times more likely to block vertically
        :type block_vert_coef: float
        """
        if last_opp_move is not None:
            action_mask[last_opp_move] *= block_vert_coef
        action = random.choices(list(range(self.boardsize)), action_mask)[0]
        return action

    def weak_rule_based_opponent(self, player):
        """Takes move for weak rule-based opponent.

        :param player: Player who we are checking, 0 or 1
        :type player: int
        """
        max_length = -1
        best_actions = []
        for action in range(self.boardsize): #GANTI
            possible, reward, ended, lengths = self.outcome(
                action, player, return_length=True
            )
            if possible and lengths.sum() > max_length:
                best_actions = []
                max_length = lengths.sum()
            if possible and lengths.sum() == max_length:
                best_actions.append(action)
        best_action = random.choice(best_actions)
        return best_action

    def strong_rule_based_opponent(self, player):
        """Takes move for strong rule-based opponent.

        :param player: Player who we are checking, 0 or 1
        :type player: int
        """

        winning_actions = []
        for action in range(self.boardsize): #GANTI
            possible, reward, ended = self.outcome(action, player)
            if possible and ended:
                winning_actions.append(action)
        if len(winning_actions) > 0:
            winning_action = random.choice(winning_actions)
            return winning_action

        opp = 1 if player == 0 else 0
        loss_avoiding_actions = []
        for action in range(self.boardsize): #GANTI
            possible, reward, ended = self.outcome(action, opp)
            if possible and ended:
                loss_avoiding_actions.append(action)
        if len(loss_avoiding_actions) > 0:
            loss_avoiding_action = random.choice(loss_avoiding_actions)
            return loss_avoiding_action

        return self.weak_rule_based_opponent(player)  # take best possible move
    
    def action_coords(self, action):
        row = action//self.num_cols
        col = (self.num_rows-1) - action%self.num_rows
        return row, col

    def outcome(self, action, player, return_length=False):
        """Takes move for weak rule-based opponent.

        :param action: Action to take in environment
        :type action: int
        :param player: Player who we are checking, 0 or 1
        :type player: int
        :param return_length: Return length of outcomes, defaults to False
        :type player: bool, optional
        """
        if np.array(self.env.env.board.squares)[action]:
            return (False, None, None) + ((None,) if return_length else ())

        row, col = self.action_coords(action)
        piece = player + 1

        directions = np.array(
            [
                [[-1, 0], [1, 0]],
                [[0, -1], [0, 1]],
                [[-1, -1], [1, 1]],
                [[-1, 1], [1, -1]],
            ]
        )

        positions = np.array([row, col]).reshape(1, 1, 1, -1) + np.expand_dims(
            directions, -2
        ) * np.arange(1, self.length).reshape(
            1, 1, -1, 1
        )

        valid_positions = np.logical_and(
            np.logical_and(
                positions[:, :, :, 0] >= 0, positions[:, :, :, 0] < self.num_rows
            ),
            np.logical_and(
                positions[:, :, :, 1] >= 0, positions[:, :, :, 1] < self.num_cols
            ),
        )

        d0 = np.where(valid_positions, positions[:, :, :, 0], 0)
        d1 = np.where(valid_positions, positions[:, :, :, 1], 0)
        board = np.array(self.env.env.board.squares).reshape(15,15)
        board_values = np.where(valid_positions, board[d0, d1], 0)

        a = (board_values == piece).astype(int)
        b = np.concatenate(
            (a, np.zeros_like(a[:, :, :1])), axis=-1
        )

        lengths = np.argmin(b, -1)

        ended = False
        for both_dir in board_values:
            line = np.concatenate((both_dir[0][::-1], [piece], both_dir[1]))
            if "".join(map(str, [piece] * self.length)) in "".join(map(str, line)):
                ended = True
                break

        temp = np.copy(self.env.env.board.squares)
        temp = np.delete(temp, action, 0)
        draw = np.all(temp)
        ended |= draw
        reward = (-1) ** (player) if ended and not draw else 0

        return (True, reward, ended) + ((lengths,) if return_length else ())

def transform_and_flip(observation, player):
    """Transforms and flips observation for input to agent's neural network.

    :param observation: Observation to preprocess
    :type observation: dict[str, np.ndarray]
    :param player: Player, 0 or 1
    :type player: int
    """
    state = observation["observation"]
    # Pre-process dimensions for PyTorch (N, C, H, W)
    state = np.moveaxis(state, [-1], [-3])
    if player == 1:
        # Swap pieces so that the agent always sees the board from the same perspective
        state[[0, 1], :, :] = state[[1, 0], :, :]
    state_flipped = np.expand_dims(np.flip(state, 2), 0)
    state = np.expand_dims(state, 0)
    return state, state_flipped

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("===== AgileRL Curriculum Learning Demo =====")

    for lesson_number in range(2, 3):
        # Load lesson for curriculum
        with open(f"/home/anangjiun/tugasakhir/agile_DQN/curriculums/LESSON{lesson_number}.yaml") as file:
            LESSON = yaml.safe_load(file)

        # Define the network configuration
        NET_CONFIG = {
            "arch": "cnn",  # Network architecture
            "h_size": [64, 64],  # Actor hidden size
            "c_size": [128],  # CNN channel size
            "k_size": [5],  # CNN kernel size
            "s_size": [1],  # CNN stride size
            "normalize": False,  # Normalize image from range [0,255] to [0,1]
        }

        # Define the initial hyperparameters
        INIT_HP = {
            "POPULATION_SIZE": 6,
            # "ALGO": "Rainbow DQN",  # Algorithm
            "ALGO": "DQN",  # Algorithm
            "DOUBLE": True,
            # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
            "BATCH_SIZE": 256,  # Batch size
            "LR": 1e-4,  # Learning rate
            "GAMMA": 0.99,  # Discount factor
            "MEMORY_SIZE": 100000,  # Max memory buffer size
            "LEARN_STEP": 1,  # Learning frequency
            "N_STEP": 1,  # Step number to calculate td error
            "PER": False,  # Use prioritized experience replay buffer
            "ALPHA": 0.6,  # Prioritized replay buffer parameter
            "TAU": 0.01,  # For soft update of target parameters
            "BETA": 0.4,  # Importance sampling coefficient
            "PRIOR_EPS": 0.000001,  # Minimum priority for sampling
            "NUM_ATOMS": 51,  # Unit number of support
            "V_MIN": 0.0,  # Minimum value of support
            "V_MAX": 200.0,  # Maximum value of support
            "WANDB": False,  # Use Weights and Biases tracking
        }

        # Define the connect four environment
        env = gomoku.env()
        env.reset()

        # Configure the algo input arguments
        state_dim = [
            env.observation_space(agent)["observation"].shape for agent in env.agents
        ]
        one_hot = False
        action_dim = [env.action_space(agent).n for agent in env.agents]
        INIT_HP["DISCRETE_ACTIONS"] = True
        INIT_HP["MAX_ACTION"] = None
        INIT_HP["MIN_ACTION"] = None

        # Warp the environment in the curriculum learning wrapper
        env = CurriculumEnv(env, LESSON)

        # Pre-process dimensions for PyTorch layers
        # We only need to worry about the state dim of a single agent
        # We flatten the 3x3x2 observation as input to the agent"s neural network
        state_dim = np.moveaxis(np.zeros(state_dim[0]), [-1], [-3]).shape
        action_dim = action_dim[0]

        # Create a population ready for evolutionary hyper-parameter optimisation
        pop = initialPopulation(
            INIT_HP["ALGO"],
            state_dim,
            action_dim,
            one_hot,
            NET_CONFIG,
            INIT_HP,
            population_size=INIT_HP["POPULATION_SIZE"],
            device=device,
        )

        # Configure the replay buffer
        field_names = ["state", "action", "reward", "next_state", "done"]
        memory = ReplayBuffer(
            action_dim=action_dim,  # Number of agent actions
            memory_size=INIT_HP["MEMORY_SIZE"],  # Max replay buffer size
            field_names=field_names,  # Field names to store in memory
            device=device,
        )

        # Instantiate a tournament selection object (used for HPO)
        tournament = TournamentSelection(
            tournament_size=2,  # Tournament selection size
            elitism=True,  # Elitism in tournament selection
            population_size=INIT_HP["POPULATION_SIZE"],  # Population size
            evo_step=1,
        )  # Evaluate using last N fitness scores

        # Instantiate a mutations object (used for HPO)
        mutations = Mutations(
            algo=INIT_HP["ALGO"],
            no_mutation=0.2,  # Probability of no mutation
            architecture=0,  # Probability of architecture mutation
            new_layer_prob=0.2,  # Probability of new layer mutation
            parameters=0.2,  # Probability of parameter mutation
            activation=0,  # Probability of activation function mutation
            rl_hp=0.2,  # Probability of RL hyperparameter mutation
            rl_hp_selection=[
                "lr",
                "learn_step",
                "batch_size",
            ],  # RL hyperparams selected for mutation
            mutation_sd=0.1,  # Mutation strength
            # Define search space for each hyperparameter
            min_lr=0.0001,
            max_lr=0.01,
            min_learn_step=1,
            max_learn_step=120,
            min_batch_size=8,
            max_batch_size=64,
            arch=NET_CONFIG["arch"],  # MLP or CNN
            rand_seed=1,
            device=device,
        )

        # Define training loop parameters
        episodes_per_epoch = 10

        # ! NOTE: Uncomment the max_episodes line below to change the number of training episodes. ! #
        # It is deliberately set low to allow testing to ensure this tutorial is sound.
        max_episodes = 50
        #max_episodes = LESSON["max_train_episodes"]  # Total episodes

        max_steps = 500  # Maximum steps to take in each episode
        evo_epochs = 20  # Evolution frequency
        evo_loop = 50  # Number of evaluation episodes
        elite = pop[0]  # Assign a placeholder "elite" agent
        epsilon = 1.0  # Starting epsilon value
        eps_end = 0.1  # Final epsilon value
        eps_decay = 0.9998  # Epsilon decays
        opp_update_counter = 0
        wb = INIT_HP["WANDB"]

        if LESSON["pretrained_path"] is not None:
            for agent in pop:
                # Load pretrained checkpoint
                agent.loadCheckpoint(LESSON["pretrained_path"])
                # Reinit optimizer for new task
                agent.lr = INIT_HP["LR"]
                agent.optimizer = torch.optim.Adam(
                    agent.actor.parameters(), lr=agent.lr
                )

        if LESSON["opponent"] == "self":
            # Create initial pool of opponents
            opponent_pool = deque(maxlen=LESSON["opponent_pool_size"])
            for _ in range(LESSON["opponent_pool_size"]):
                opp = copy.deepcopy(pop[0])
                opp.actor.load_state_dict(pop[0].actor.state_dict())
                opp.actor.eval()
                opponent_pool.append(opp)

        # Perform buffer and agent warmups if desired
        if LESSON["buffer_warm_up"]:
            warm_up_opponent = Opponent(env, difficulty=LESSON["warm_up_opponent"])
            memory = env.fill_replay_buffer(
                memory, warm_up_opponent
            )  # Fill replay buffer with transitions
            if LESSON["agent_warm_up"] > 0:
                print("Warming up agents ...")
                agent = pop[0]
                # Train on randomly collected samples
                for epoch in trange(LESSON["agent_warm_up"]):
                #for epoch in trange(20):
                    experiences = memory.sample(agent.batch_size)
                    agent.learn(experiences)
                pop = [agent.clone() for _ in pop]
                elite = agent
                print("Agent population warmed up.")

        if max_episodes > 0:
            if wb:
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="AgileRL",
                    name="{}-EvoHPO-{}-{}Opposition-CNN-{}".format(
                        "gomoku_v0",
                        INIT_HP["ALGO"],
                        LESSON["opponent"],
                        datetime.now().strftime("%m%d%Y%H%M%S"),
                    ),
                    # track hyperparameters and run metadata
                    config={
                        "algo": "Evo HPO DQN",
                        "env": "gomoku_v0",
                        "INIT_HP": INIT_HP,
                        "lesson": LESSON,
                    },
                )

        total_steps = 0
        total_episodes = 0
        pbar = trange(int(max_episodes / episodes_per_epoch))
        boardsize = 15**2 - 1

        # Training loop
        for idx_epi in pbar:
            turns_per_episode = []
            train_actions_hist = [0] * action_dim
            for agent in pop:  # Loop through population
                for episode in range(episodes_per_epoch):
                    print("checkpoint 2")
                    env.reset()  # Reset environment at start of episode
                    observation, env_reward, done, truncation, _ = env.last()

                    (
                        p1_state,
                        p1_state_flipped,
                        p1_action,
                        p1_next_state,
                        p1_next_state_flipped,
                    ) = (None, None, None, None, None)

                    if LESSON["opponent"] == "self":
                        # Randomly choose opponent from opponent pool if using self-play
                        opponent = random.choice(opponent_pool)
                    else:
                        # Create opponent of desired difficulty
                        opponent = Opponent(env, difficulty=LESSON["opponent"])

                    # Randomly decide whether agent will go first or second
                    if random.random() > 0.5:
                        opponent_first = False
                    else:
                        opponent_first = True

                    score = 0
                    turns = 0  # Number of turns counter

                    for idx_step in range(max_steps):
                        
                        # Player 0"s turn
                        #p0_action_mask64 = np.array([int(i) for i in observation['action_mask']])
                        p0_action_mask8 = observation["action_mask"]
                        p0_state, p0_state_flipped = transform_and_flip(
                            observation, player=0
                        )

                        if opponent_first:
                            if LESSON["opponent"] == "self":
                                p0_action = opponent.getAction(
                                    p0_state, 0, p0_action_mask8
                                )[0]
                            elif LESSON["opponent"] == "random":
                                p0_action = opponent.getAction(
                                    p0_action_mask8, p1_action, LESSON["block_vert_coef"]
                                )
                            else:
                                p0_action = opponent.getAction(player=0)
                        else:
                            p0_action = agent.getAction(
                                p0_state, epsilon, p0_action_mask8
                            )[
                                0
                            ]  # Get next action from agent
                            train_actions_hist[p0_action] += 1

                        env.step(p0_action)  # Act in environment
                        observation, env_reward, done, truncation, _ = env.last()
                        p0_next_state, p0_next_state_flipped = transform_and_flip(
                            observation, player=0
                        )

                        if not opponent_first:
                            score += env_reward
                        turns += 1

                        # Check if game is over (Player 0 win)
                        if done or truncation:
                            reward = env.reward(done=True, player=0)
                            memory.save2memoryVectEnvs(
                                np.concatenate(
                                    (
                                        p0_state,
                                        p1_state,
                                        p0_state_flipped,
                                        p1_state_flipped,
                                    )
                                ),
                                [p0_action, p1_action, boardsize - p0_action, boardsize - p1_action],
                                [
                                    reward,
                                    LESSON["rewards"]["lose"],
                                    reward,
                                    LESSON["rewards"]["lose"],
                                ],
                                np.concatenate(
                                    (
                                        p0_next_state,
                                        p1_next_state,
                                        p0_next_state_flipped,
                                        p1_next_state_flipped,
                                    )
                                ),
                                [done, done, done, done],
                            )
                        else:  # Play continues
                            if p1_state is not None:
                                reward = env.reward(done=False, player=1)
                                memory.save2memoryVectEnvs(
                                    np.concatenate((p1_state, p1_state_flipped)),
                                    [p1_action, boardsize - p1_action],
                                    [reward, reward],
                                    np.concatenate(
                                        (p1_next_state, p1_next_state_flipped)
                                    ),
                                    [done, done],
                                )

                            # Player 1"s turn
                            #p1_action_mask64 = np.array([int(i) for i in observation['action_mask']])
                            p1_action_mask8 = observation["action_mask"]
                            p1_state, p1_state_flipped = transform_and_flip(
                                observation, player=1
                            )

                            if not opponent_first:
                                if LESSON["opponent"] == "self":
                                    p1_action = opponent.getAction(
                                        p1_state, 0, p1_action_mask8
                                    )[0]
                                elif LESSON["opponent"] == "random":
                                    p1_action = opponent.getAction(
                                        p1_action_mask8,
                                        p0_action,
                                        LESSON["block_vert_coef"],
                                    )
                                else:
                                    p1_action = opponent.getAction(player=1)
                            else:
                                p1_action = agent.getAction(
                                    p1_state, epsilon, p1_action_mask8
                                )[
                                    0
                                ]  # Get next action from agent
                                train_actions_hist[p1_action] += 1

                            env.step(p1_action)  # Act in environment
                            observation, env_reward, done, truncation, _ = env.last()
                            p1_next_state, p1_next_state_flipped = transform_and_flip(
                                observation, player=1
                            )

                            if opponent_first:
                                score += env_reward
                            turns += 1

                            # Check if game is over (Player 1 win)
                            if done or truncation:
                                reward = env.reward(done=True, player=1)
                                memory.save2memoryVectEnvs(
                                    np.concatenate(
                                        (
                                            p0_state,
                                            p1_state,
                                            p0_state_flipped,
                                            p1_state_flipped,
                                        )
                                    ),
                                    [
                                        p0_action,
                                        p1_action,
                                        boardsize - p0_action,
                                        boardsize - p1_action,
                                    ],
                                    [
                                        LESSON["rewards"]["lose"],
                                        reward,
                                        LESSON["rewards"]["lose"],
                                        reward,
                                    ],
                                    np.concatenate(
                                        (
                                            p0_next_state,
                                            p1_next_state,
                                            p0_next_state_flipped,
                                            p1_next_state_flipped,
                                        )
                                    ),
                                    [done, done, done, done],
                                )

                            else:  # Play continues
                                reward = env.reward(done=False, player=0)
                                memory.save2memoryVectEnvs(
                                    np.concatenate((p0_state, p0_state_flipped)),
                                    [p0_action, boardsize - p0_action],
                                    [reward, reward],
                                    np.concatenate(
                                        (p0_next_state, p0_next_state_flipped)
                                    ),
                                    [done, done],
                                )
                        # Learn according to learning frequency
                        if (memory.counter % agent.learn_step == 0) and (
                            len(memory) >= agent.batch_size
                        ):
                            # Sample replay buffer
                            # Learn according to agent"s RL algorithm
                            experiences = memory.sample(agent.batch_size)
                            agent.learn(experiences)
                        
                        # Stop episode if any agents have terminated
                        if done or truncation:
                            break

                    total_steps += idx_step + 1
                    total_episodes += 1
                    turns_per_episode.append(turns)
                    # Save the total episode reward
                    agent.scores.append(score)
                    
                    if LESSON["opponent"] == "self":
                        if (total_episodes % LESSON["opponent_upgrade"] == 0) and (
                            (idx_epi + 1) > evo_epochs
                        ):
                            elite_opp, _, _ = tournament._elitism(pop)
                            elite_opp.actor.eval()
                            opponent_pool.append(elite_opp)
                            opp_update_counter += 1
                    print("checkpoint 2c", turns)
                print("checkpoint 4")
                # Update epsilon for exploration
                epsilon = max(eps_end, epsilon * eps_decay)

            mean_turns = np.mean(turns_per_episode)

            # Now evolve population if necessary
            if (idx_epi + 1) % evo_epochs == 0:
                # Evaluate population vs random actions
                fitnesses = []
                win_rates = []
                eval_actions_hist = [0] * action_dim  # Eval actions histogram
                eval_turns = 0  # Eval turns counter
                for agent in pop:
                    with torch.no_grad():
                        rewards = []
                        for i in range(evo_loop):
                            env.reset()  # Reset environment at start of episode
                            observation, reward, done, truncation, _ = env.last()

                            player = -1  # Tracker for which player"s turn it is

                            # Create opponent of desired difficulty
                            opponent = Opponent(env, difficulty=LESSON["eval_opponent"])

                            # Randomly decide whether agent will go first or second
                            if random.random() > 0.5:
                                opponent_first = False
                            else:
                                opponent_first = True

                            score = 0

                            for idx_step in range(max_steps):
                                #action_mask = np.array([int(i) for i in observation['action_mask']])
                                action_mask = observation['action_mask']
                                if player < 0:
                                    if opponent_first:
                                        if LESSON["eval_opponent"] == "random":
                                            action = opponent.getAction(action_mask)
                                        else:
                                            action = opponent.getAction(player=0)
                                    else:
                                        state = np.moveaxis(
                                            observation["observation"], [-1], [-3]
                                        )
                                        state = np.expand_dims(state, 0)
                                        action = agent.getAction(state, 0, action_mask)[
                                            0
                                        ]  # Get next action from agent
                                        eval_actions_hist[action] += 1
                                if player > 0:
                                    if not opponent_first:
                                        if LESSON["eval_opponent"] == "random":
                                            action = opponent.getAction(action_mask)
                                        else:
                                            action = opponent.getAction(player=1)
                                    else:
                                        state = np.moveaxis(
                                            observation["observation"], [-1], [-3]
                                        )
                                        state[[0, 1], :, :] = state[[0, 1], :, :]
                                        state = np.expand_dims(state, 0)
                                        action = agent.getAction(state, 0, action_mask)[
                                            0
                                        ]  # Get next action from agent
                                        eval_actions_hist[action] += 1

                                env.step(action)  # Act in environment
                                observation, reward, done, truncation, _ = env.last()

                                if (player > 0 and opponent_first) or (
                                    player < 0 and not opponent_first
                                ):
                                    score += reward

                                eval_turns += 1

                                if done or truncation:
                                    break

                                player *= -1

                            rewards.append(score)
                    mean_fit = np.mean(rewards)
                    agent.fitness.append(mean_fit)
                    fitnesses.append(mean_fit)

                eval_turns = eval_turns / len(pop) / evo_loop

                pbar.set_postfix_str(
                    f"    Train Mean Score: {np.mean(agent.scores[-episodes_per_epoch:])}   Train Mean Turns: {mean_turns}   Eval Mean Fitness: {np.mean(fitnesses)}   Eval Best Fitness: {np.max(fitnesses)}   Eval Mean Turns: {eval_turns}   Total Steps: {total_steps}"
                )
                pbar.update(0)

                # Format action histograms for visualisation
                train_actions_hist = [
                    freq / sum(train_actions_hist) for freq in train_actions_hist
                ]
                eval_actions_hist = [
                    freq / sum(eval_actions_hist) for freq in eval_actions_hist
                ]
                train_actions_dict = {
                    f"train/action_{index}": action
                    for index, action in enumerate(train_actions_hist)
                }
                eval_actions_dict = {
                    f"eval/action_{index}": action
                    for index, action in enumerate(eval_actions_hist)
                }

                if wb:
                    wandb_dict = {
                        "global_step": total_steps,
                        "train/mean_score": np.mean(agent.scores[-episodes_per_epoch:]),
                        "train/mean_turns_per_game": mean_turns,
                        "train/epsilon": epsilon,
                        "train/opponent_updates": opp_update_counter,
                        "eval/mean_fitness": np.mean(fitnesses),
                        "eval/best_fitness": np.max(fitnesses),
                        "eval/mean_turns_per_game": eval_turns,
                    }
                    wandb_dict.update(train_actions_dict)
                    wandb_dict.update(eval_actions_dict)
                    wandb.log(wandb_dict)

                # Tournament selection and population mutation
                elite, pop = tournament.select(pop)
                pop = mutations.mutation(pop)

        if max_episodes > 0:
            if wb:
                wandb.finish()

        # Save the trained agent
        save_path = LESSON["save_path"]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        elite.saveCheckpoint(save_path)
        print(f"Elite agent saved to '{save_path}'.")