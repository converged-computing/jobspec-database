import os
import random

import jax

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

from jaxrl.agents import SACLearner
from jaxrl.datasets.replay_buffer import ReplayBuffer
from jaxrl.datasets.dataset import Batch
from jaxrl.evaluation import evaluate
from jaxrl.utils import make_env
from env_model.loss_functions import (
    make_loss,
    quadratic_vagram_loss,
    mse_loss,
    vagram_joint_loss,
    vagram_loss,
    vagram_no_bounds,
)
from env_model.model_train import train_model
from env_model.nn_modules import init_model

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "Pendulum-v1", "Environment name.")
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")
flags.DEFINE_string("model_loss_fn", "mse", "Loss function to update model")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 1000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("updates_per_step", 20, "Gradient updates per step.")
flags.DEFINE_integer(
    "model_update_interval", 250, "Number of training steps until model is updated"
)
flags.DEFINE_integer("model_hidden_size", 128, "Gradient updates per step.")
flags.DEFINE_integer("model_steps", 5000, "Gradient updates per step.")
flags.DEFINE_integer("model_batch_size", 128, "Gradient updates per step.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer("num_model_samples", int(50000), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(4e3), "Number of training steps to start training."
)
flags.DEFINE_float("model_lr", 1e-3, "Learning rate for the model.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
config_flags.DEFINE_config_file(
    "config",
    "configs/sac_default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def main(_):
    key = jax.random.PRNGKey(FLAGS.seed)

    summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir, "tb", str(FLAGS.seed)))

    if FLAGS.save_video:
        video_train_folder = os.path.join(FLAGS.save_dir, "video", "train")
        video_eval_folder = os.path.join(FLAGS.save_dir, "video", "eval")
    else:
        video_train_folder = None
        video_eval_folder = None

    if FLAGS.model_loss_fn == "mse":
        loss_fn = mse_loss
    elif FLAGS.model_loss_fn == "vagram":
        loss_fn = vagram_loss
    elif FLAGS.model_loss_fn == "vagram_quadratic":
        loss_fn = quadratic_vagram_loss
    elif FLAGS.model_loss_fn == "vagram_full":
        loss_fn = vagram_joint_loss
    elif FLAGS.model_loss_fn == "vagram_no_bound":
        loss_fn = vagram_no_bounds
    else:
        raise ValueError(f"Unknown loss function: {FLAGS.model_loss_fn}")

    env = make_env(FLAGS.env_name, FLAGS.seed, video_train_folder)
    eval_env = make_env(FLAGS.env_name, FLAGS.seed + 42, video_eval_folder)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    kwargs = dict(FLAGS.config)
    algo = kwargs.pop("algo")
    replay_buffer_size = kwargs.pop("replay_buffer_size")
    agent = SACLearner(
        FLAGS.seed,
        env.observation_space.sample()[np.newaxis],
        env.action_space.sample()[np.newaxis],
        **kwargs,
    )
    key, init_key = jax.random.split(key)
    model_state, model_network = init_model(
        env.observation_space,
        env.action_space,
        FLAGS.model_hidden_size,
        FLAGS.model_lr,
        init_key,
    )

    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, replay_buffer_size or FLAGS.max_steps
    )

    eval_returns = []
    observation, done = env.reset(), False
    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if not done or "TimeLimit.truncated" in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(
            observation, action, reward, mask, float(done), next_observation
        )

        observation = next_observation

        if done:
            observation, done = env.reset(), False
            for k, v in info["episode"].items():
                summary_writer.add_scalar(
                    f"training/{k}", v, info["total"]["timesteps"]
                )

            if "is_success" in info:
                summary_writer.add_scalar(
                    f"training/success", info["is_success"], info["total"]["timesteps"]
                )

        if i >= FLAGS.start_training:
            if i % FLAGS.model_update_interval == 0 or i == FLAGS.start_training:

                def vf(x):
                    return np.prod(np.sin(x))

                loss_function = make_loss(
                    loss_fn,
                    model_network,
                    agent.compile_state_value_function(),
                )
                model_replay_buffer = ReplayBuffer(
                    env.observation_space,
                    env.action_space,
                    FLAGS.num_model_samples,
                )
                key, model_train_key = jax.random.split(key, 2)
                model_state, metrics = train_model(
                    model_state,
                    replay_buffer,
                    loss_function,
                    FLAGS.model_batch_size,
                    model_train_key,
                )
                model_fn = lambda s, a: model_state.apply_fn(
                    {"params": model_state.params},
                    jax.numpy.concatenate([s, a], axis=1),
                )

                # batch = replay_buffer.sample(FLAGS.num_model_samples)
                # actions = agent.sample_actions(batch.observations)
                # next_states, rewards = model_state.apply_fn(
                #     {"params": model_state.params},
                #     jax.numpy.concatenate([batch.observations, actions], axis=1),
                # )
                # ensemble_indices = np.random.choice(
                #     8, size=(FLAGS.num_model_samples)
                # ).reshape(1, -1, 1)
                # next_states = jax.numpy.take_along_axis(
                #     next_states, ensemble_indices, axis=0
                # )[0]

                # for observation, action, reward, next_observation in zip(
                #     batch.observations, actions, rewards, next_states
                # ):
                #     # TODO: Check that mask and done is set correctly
                #     model_replay_buffer.insert(
                #         observation, action, reward, 1.0, 0.0, next_observation
                #     )

            for _ in range(FLAGS.updates_per_step):
                buffer_name = np.random.choice(["model", "env"], p=[0.95, 0.05])
                batch = replay_buffer.sample(FLAGS.batch_size)
                if buffer_name == "model":
                    actions = agent.sample_actions(batch.observations)
                    next_states, rewards = jax.jit(model_fn)(
                        batch.observations, actions
                    )
                    ensemble_indices = np.random.choice(
                        8, size=(FLAGS.batch_size)
                    ).reshape(1, -1, 1)
                    next_states = jax.numpy.take_along_axis(
                        next_states, ensemble_indices, axis=0
                    )[0]
                    batch = Batch(
                        observations=batch.observations,
                        actions=actions,
                        rewards=rewards,
                        masks=batch.masks,
                        next_observations=next_states,
                    )
                update_info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    summary_writer.add_scalar(f"training/{k}", v, i)
                summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(
                    f"evaluation/average_{k}s", v, info["total"]["timesteps"]
                )
            summary_writer.flush()

            eval_returns.append((info["total"]["timesteps"], eval_stats["return"]))
            np.savetxt(
                os.path.join(FLAGS.save_dir, f"{FLAGS.model_loss_fn}_{FLAGS.seed}.txt"),
                eval_returns,
                fmt=["%d", "%.1f"],
            )


if __name__ == "__main__":
    app.run(main)
