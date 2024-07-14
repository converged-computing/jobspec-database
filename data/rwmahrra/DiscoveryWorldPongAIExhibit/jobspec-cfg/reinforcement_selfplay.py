
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

import os
from exhibit.train import simulator
from exhibit.shared.utils import save_video, plot_loss, plot_score, plot_duration
from exhibit.shared.config import Config
from exhibit.ai.model import PGAgent
from visualizer import get_weight_image
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

"""
This file is the driver for training a new DRL pong model.
It brings together the following elements:

* The environment simulator (either the custom one found in pong.py or the Atari emulator provided by OpenAI Gym)
  Both environments are wrapped by the interface in simulator.py
* The two agents (some combination of human-controlled, DRL, and hard-coded agents found in player.py)
The level of abstraction in this file is pretty high, and it really only exists to further abstract the training
process into a few environmental and training hyperparameters that are easy to experiment with and to provide
convenient monitoring and graphing of the training process.
"""

GAME_BATCH = 1
EPISODES = 30000
MODE = Config.instance().CUSTOM
LEARNING_RATE = 0.001
DENSE_STRUCTURE = (200,)
ALWAYS_FOLLOW = False
PARALLELIZE = False

if __name__ == "__main__":
    # Ensure directory safety
    os.makedirs("models/bottom", exist_ok=True)
    os.makedirs("models/top", exist_ok=True)
    os.makedirs("analytics", exist_ok=True)
    os.makedirs("analytics/plots", exist_ok=True)

    # Initialize for checks & scope
    start_index = None

    config = Config.instance()

    # Set constants for custom env
    action_size = config.CUSTOM_ACTION_SIZE
    state_size = config.CUSTOM_STATE_SIZE
    state_shape = config.CUSTOM_STATE_SHAPE
    #agent_l = BotPlayer(left=True,
    #                    always_follow=ALWAYS_FOLLOW) if MODE == Config.CUSTOM else None  # Default to bot, override with model if needed
    # Switch out for interactive session (against human)
    # from exhibit.game.player import HumanPlayer
    # agent_top = HumanPlayer(up='w', down='s')

    # Init agent
    if MODE == config.HIT_PRACTICE:
        agent_bottom = None
    else:
        agent_bottom = PGAgent(state_size, action_size, name="agent_bottom", learning_rate=LEARNING_RATE, structure=DENSE_STRUCTURE)
        agent_bottom.load("./validation/smoothreward_s6_f5_d3_22850.h5")

    agent_top = PGAgent(state_size, action_size, name="agent_top", learning_rate=LEARNING_RATE, structure=DENSE_STRUCTURE)
    agent_top.load("./validation/smoothreward_s6_f5_d3_22850.h5")

    # Type checks for convenience later
    top_is_model = type(agent_top) == PGAgent
    bottom_is_model = type(agent_bottom) == PGAgent

    episode = 0

    # Optional checkpoint loading
    if start_index is not None:
        episode = start_index
        if bottom_is_model: agent_bottom.load(f'./models/bottom/{start_index}.h5')
        agent_top.load(f'./models/top/{start_index}.h5')

    # Store neuron images for fun
    neuron_states = []
    # Train loop
    for episode in tqdm(range(EPISODES)):
        episode += 1
        bottom_path = None
        top_path = None
        if bottom_is_model:
            bottom_path = './models/bottom/latest.h5'
            agent_bottom.save(bottom_path)
        if top_is_model:
            top_path = './models/top/latest.h5'
            agent_top.save(top_path)

        states_bundle, left, right, meta = simulator.simulate_game(
            config, env_type=MODE, left=agent_bottom, right=agent_top, batch=GAME_BATCH)
        states_bottom, states_top = states_bundle
        render_states, model_states, (score_l, score_r) = meta
        actions, probs, rewards = right

        if top_is_model:
            agent_top.train(states_top, *right)
        if bottom_is_model:
            agent_bottom.train(states_bottom, *left)

        neuron_states.append(get_weight_image(agent_top.train_model, size=state_shape))
        if episode == 1 or episode % 50 == 0:
            save_video(render_states, f'./analytics/{episode}.mp4')
            plot_loss(f'./analytics/plots/loss_{episode}.png', include_left=True)
            plot_score(f'./analytics/plots/score_{episode}.png')
            plot_duration(f'./analytics/plots/duration_{episode}.png')
            if bottom_is_model: agent_bottom.save(f'./models/bottom/{episode}.h5')
            if top_is_model: agent_top.save(f'./models/top/{episode}.h5')
        if episode == EPISODES:
            if top_is_model: save_video(neuron_states, f'./analytics/{episode}_weights0.mp4', fps=60)
            exit(0)
