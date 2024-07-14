import itertools as it
import os
from time import time, sleep
import argparse

import torch
import vizdoom as vzd
import skimage
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

from agents import DQNAgent, DoubleDQNAgent, DuelDQNAgent


def preprocess(img):
    """Down samples image to resolution"""
    img = skimage.transform.resize(img, (30, 45))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img


def create_game_environment(config_file_path):
    print("Creating game environment ...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    print("Game environment initialized ...")
    return game


def test_agent(game, agent, actions, frame_repeat, test_episodes_per_epoch=10):
    """Runs a test_episodes_per_epoch episodes and prints the result"""
    print("\nTesting...")
    test_scores = []
    for _ in trange(test_episodes_per_epoch):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            best_action_index = agent.get_action(state)
            game.make_action(actions[best_action_index], frame_repeat)
        reward = game.get_total_reward()
        test_scores.append(reward)

    test_scores = np.array(test_scores)
    return test_scores


def train_agent(game, agent, actions, num_epochs, frame_repeat, steps_per_epoch, save_model, model_path, model_name):
    """
    Trains the DQN Agent by running num_epochs of training episodes.
    Skip frame_repeat number of frames after each action.
    """

    start_time = time()
    train_reward_scores = []
    test_reward_scores = []


    for epoch in range(num_epochs):
        game.new_episode()
        train_scores = []
        global_step = 0

        print("\nEpoch #" + str(epoch + 1))

        for _ in trange(steps_per_epoch):
            state = preprocess(game.get_state().screen_buffer)
            action = agent.get_action(state)
            reward = game.make_action(actions[action], frame_repeat)
            done = game.is_episode_finished()

            if not done:
                next_state = preprocess(game.get_state().screen_buffer)
            else:
                next_state = np.zeros((1, 30, 45)).astype(np.float32)

            agent.append_memory(state, action, reward, next_state, done)

            if global_step > agent.batch_size:
                agent.train()

            if done:
                train_scores.append(game.get_total_reward())
                game.new_episode()

            global_step += 1

        if model_name == 'duel_dqn':
            agent.update_target_net()

        train_scores = np.array(train_scores)
        print(
            "Results (Train): mean: {:.1f} +/- {:.1f},".format(
                train_scores.mean(), train_scores.std()
            ),
            "min: %.1f," % train_scores.min(),
            "max: %.1f," % train_scores.max(),
        )
        train_reward_scores.append(train_scores.mean())

        test_scores = test_agent(game, agent, actions, frame_repeat)
        print(
            "Results (Test): mean: {:.1f} +/- {:.1f},".format(
                test_scores.mean(), test_scores.std()
            ),
            "min: %.1f" % test_scores.min(),
            "max: %.1f" % test_scores.max(),
        )
        test_reward_scores.append(test_scores.mean())

        if save_model:
            print("Saving the network weights to:", model_path)
            torch.save(agent.q_net, model_path)
        print("Total elapsed time: %.2f minutes" %
              ((time() - start_time) / 60.0))

    plot(train_reward_scores, test_reward_scores, model_name,
         x=[i for i in range(num_epochs)], y1_label="Train", y2_label="Test")
    game.close()
    return agent, game


def plot(y1, y2, name, x, y1_label, y2_label):
    plt.figure()
    if y1 is not None:
        plt.plot(x, y1, label=y1_label)
    if y2 is not None:
        plt.plot(x, y2, label=y2_label)
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Reward Score")
    plt.savefig("plots/" + name + ".jpg")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments to customize DQN training"
    )

    # Model and Environment
    parser.add_argument("--model", default="ddqn", type=str)
    parser.add_argument("--scenario", default="simpler_basic", type=str)
    parser.add_argument("--checkpoint", default="doom.pth", type=str)

    # Hyperparameters
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--discount-factor", default=0.99, type=float)
    parser.add_argument("--memory-size", default=10000, type=int)
    parser.add_argument("--frame-repeat", default=12, type=int)
    parser.add_argument("--steps-per-epoch", default=2000, type=int)
    parser.add_argument("--epsilon-decay", default=0.99, type=float)
    parser.add_argument("--num-epochs", default=50, type=int)

    # Others
    parser.add_argument("--skip-learning", default=False, type=bool)
    parser.add_argument("--load-model", default=False, type=bool)

    return parser.parse_args()


def run(args):
    scenario_file = args.scenario + ".cfg"

    config_file_path = os.path.join(vzd.scenarios_path, scenario_file)
    game = create_game_environment(config_file_path)
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]
    action_size = len(actions)

    # Set the hyperparameters
    batch_size = args.batch_size
    lr = args.lr
    discount_factor = args.discount_factor
    memory_size = args.memory_size
    frame_repeat = args.frame_repeat
    steps_per_epoch = args.steps_per_epoch
    epsilon_decay = args.epsilon_decay
    model_name = args.model
    skip_learning = args.skip_learning
    load_model = args.load_model
    num_epochs = args.num_epochs
    checkpoint = args.checkpoint
    print("\nHyperparameters")
    print(f"\tbatch_size = {batch_size}")
    print(f"\tlr = {lr}")
    print(f"\tdiscount_factor = {discount_factor}")
    print(f"\tmemory_size = {memory_size}")
    print(f"\tframe_repeat = {frame_repeat}")
    print(f"\tsteps_per_epoch = {steps_per_epoch}")
    print(f"\tepsilon_decay = {epsilon_decay}")
    print(f"\tnum_epochs = {num_epochs}")
    print(f"\tmodel = {model_name}")
    print(f"\tskip_learning = {skip_learning}")
    print(f"\tload_model = {load_model}")
    print(f"\tcheckpoint = {checkpoint}\n")

    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    print(f"Using device={device} ...")

    # Initialize our agent with the set parameters
    if model_name == 'dqn':
        agent = DQNAgent(
            action_size=action_size,
            lr=lr,
            batch_size=batch_size,
            memory_size=memory_size,
            discount_factor=discount_factor,
            load_model=args.load_model,
            device=device,
            epsilon_decay=epsilon_decay,
            model_savefile=checkpoint
        )
    elif model_name == 'double_dqn':
        agent = DoubleDQNAgent(
            action_size=action_size,
            lr=lr,
            batch_size=batch_size,
            memory_size=memory_size,
            discount_factor=discount_factor,
            load_model=args.load_model,
            device=device,
            epsilon_decay=epsilon_decay,
            model_savefile=checkpoint
        )
    else:
        agent = DuelDQNAgent(
            action_size=action_size,
            lr=lr,
            batch_size=batch_size,
            memory_size=memory_size,
            discount_factor=discount_factor,
            load_model=args.load_model,
            device=device,
            epsilon_decay=epsilon_decay,
            model_savefile=checkpoint
        )

    # Run the training for the set number of epochs
    if not skip_learning:
        agent, game = train_agent(
            game,
            agent,
            actions,
            num_epochs=num_epochs,
            frame_repeat=frame_repeat,
            steps_per_epoch=steps_per_epoch,
            save_model=True,
            model_path="checkpoints/" + checkpoint,
            model_name=model_name,
        )
        print("======================================")
        print("Training finished. It's time to watch!")

    # Reinitialize the game with window visible
    game.close()
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.init()

    scores = []

    for _ in range(100):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            best_action_index = agent.get_action(state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(12):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        scores.append(score)
        print("Total score: ", score)

    plot(y1=scores, y2=None, name=model_name + "_validation",
         x=[i for i in range(100)], y1_label="Validation", y2_label=None)


if __name__ == "__main__":
    args = parse_args()
    run(args)
