#!/usr/bin/env python3
import os
import time
import argparse
import logging
import numpy as np
import collections

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torch.nn.functional as F

from tensorboardX import SummaryWriter

from libcube import cubes
from libcube import model
from libcube import conf

import validation

log = logging.getLogger("train")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ini", required=True, help="Ini file to use for this run")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    config = conf.Config(args.ini)
    device = torch.device("cuda" if config.train_cuda else "cpu")

    name = config.train_name(suffix=args.name)
    writer = SummaryWriter(comment="-" + name + '_' + os.getenv('name', ''))
    save_path = os.path.join("saves", name)
    os.makedirs(save_path)

    cube_env = cubes.get(config.cube_type)
    assert isinstance(cube_env, cubes.CubeEnv)
    log.info("Selected cube: %s", cube_env)
    value_targets_method = model.ValueTargetsMethod(config.train_value_targets_method)

    net = model.Net(cube_env.encoded_shape, len(cube_env.action_enum)).to(device)
    print(net)
    # opt = optim.Adam(net.parameters(), lr=config.train_learning_rate)
    opt = optim.RMSprop(net.parameters(), lr=config.train_learning_rate)
    sched = scheduler.StepLR(opt, 1, gamma=config.train_lr_decay_gamma) if config.train_lr_decay_enabled else None

    step_idx = 0
    buf_policy_loss, buf_value_loss, buf_loss = [], [], []
    buf_policy_loss_raw, buf_value_loss_raw, buf_loss_raw = [], [], []
    buf_mean_values = []
    buf_weights, buf_weights_min, buf_weights_max = [], [], []
    ts = time.time()
    best_loss = None

    weight_decay_mult = 1.0

    if config.iterative_scramble_deepening:
        current_scramble_depth = 1
    else:
        current_scramble_depth = config.train_scramble_depth

    log.info("Generate scramble buffer...")
    scramble_buf = collections.deque(maxlen=config.scramble_buffer_batches*config.train_batch_size)
    scramble_buf.extend(model.make_scramble_buffer(cube_env, config.train_batch_size*config.scramble_buffer_batches, current_scramble_depth))
    log.info("Generated buffer of size %d", len(scramble_buf))

    while True:
        if config.train_lr_decay_enabled and step_idx % config.train_lr_decay_batches == 0:
            sched.step()
            log.info("LR decrease to %s", sched.get_lr()[0])
            writer.add_scalar("lr", sched.get_lr()[0], step_idx)

        if config.decay_inv_weights and step_idx % 5000 == 0:
            weight_decay_mult *= 0.95
            log.info("Weight decay multiplier decreased to %s", weight_decay_mult)

        step_idx += 1
        x_t, weights_t, y_policy_t, y_value_t = model.sample_batch(
            scramble_buf, net, device, config.train_batch_size, value_targets_method)

        opt.zero_grad()
        # get outputs
        policy_out_t, value_out_t = net(x_t)
        value_out_t = value_out_t.squeeze(-1)

        # make weights less harsh
        weights_t = weights_t.sqrt()

        # decay weights
        # decay_weights = 1 - (1 - weights_t) * 2.71**(-step_idx / 3000)
        decay_weights = 1 - (1 - weights_t) * weight_decay_mult

        # value loss
        value_loss_t = (value_out_t - y_value_t)**2
        value_loss_raw_t = value_loss_t.mean()
        # value weighting
        if config.weight_samples:
            # value_loss_t *= weights_t
            value_loss_t *= decay_weights
        value_loss_t = value_loss_t.mean()
        # policy loss
        policy_loss_t = F.cross_entropy(policy_out_t, y_policy_t, reduction='none')
        policy_loss_raw_t = policy_loss_t.mean()
        # policy weighting
        if config.weight_samples:
            # policy_loss_t *= weights_t
            policy_loss_t *= decay_weights
        policy_loss_t = policy_loss_t.mean()
        # total loss
        loss_raw_t = policy_loss_raw_t + value_loss_raw_t
        loss_t = value_loss_t + 0.3*policy_loss_t
        # backprop
        loss_t.backward()
        opt.step()

        # save data
        buf_weights.append(decay_weights.mean().item())
        buf_weights_min.append(decay_weights.min().item())
        buf_weights_max.append(decay_weights.max().item())

        buf_mean_values.append(value_out_t.mean().item())
        buf_policy_loss.append(policy_loss_t.item())
        buf_value_loss.append(value_loss_t.item())
        buf_loss.append(loss_t.item())
        buf_loss_raw.append(loss_raw_t.item())
        buf_value_loss_raw.append(value_loss_raw_t.item())
        buf_policy_loss_raw.append(policy_loss_raw_t.item())

        if step_idx % config.validation_iters == 0:
            print(" ==== Validation ==== ")

            def validation_stats(scramble_depth):
                solutions, iterations = validation.solve_random_cubes(cube_env,
                    scramble_depth=scramble_depth,
                    amount=100,
                    max_iterations=200,
                    net=net, device=device)

                solved = [s for s in solutions if s]
                lengths = [len(s) for s in solved]
                iterations_not_none = [i for i in iterations if i]

                pct_solved = len([s for s in solved]) / len(solutions) * 100
                if len(iterations_not_none) > 0:
                    iterations_75_percentile = np.percentile(iterations_not_none, [75])[0]
                else:
                    iterations_75_percentile = -1

                np.set_printoptions(formatter={'float': '{: 0.1f}'.format})
                print('depth:', scramble_depth)
                print('percent solved:', pct_solved)
                print('iterations per solve:', np.percentile([i for i in iterations if i], [0, 25, 50, 75, 100]) if len(iterations_not_none) > 0 else None)
                print('solution length:', np.percentile(lengths, [0, 25, 50, 75, 100]) if len(lengths) > 0 else None)

                writer.add_scalar(f"validation_pct_solved_{scramble_depth}", pct_solved, step_idx)
                writer.add_scalar(f"iterations_75_percentile_{scramble_depth}", iterations_75_percentile, step_idx)

            validation_stats(scramble_depth=7)
            validation_stats(scramble_depth=10)
            validation_stats(scramble_depth=20)

        if config.train_report_batches is not None and step_idx % config.train_report_batches == 0:
            m_policy_loss = np.mean(buf_policy_loss)
            m_value_loss = np.mean(buf_value_loss)
            m_loss = np.mean(buf_loss)
            buf_value_loss.clear()
            buf_policy_loss.clear()
            buf_loss.clear()

            m_policy_loss_raw = np.mean(buf_policy_loss_raw)
            m_value_loss_raw = np.mean(buf_value_loss_raw)
            m_loss_raw = np.mean(buf_loss_raw)
            buf_value_loss_raw.clear()
            buf_policy_loss_raw.clear()
            buf_loss_raw.clear()

            m_weights = np.mean(buf_weights)
            m_weights_min = np.mean(buf_weights_min)
            m_weights_max = np.mean(buf_weights_max)
            buf_weights.clear()
            buf_weights_min.clear()
            buf_weights_max.clear()

            m_values = np.mean(buf_mean_values)
            buf_mean_values.clear()

            dt = time.time() - ts
            ts = time.time()
            speed = config.train_batch_size * config.train_report_batches / dt
            log.info("%d: p_loss=%.3e, v_loss=%.3e, loss=%.3e, speed=%.1f cubes/s",
                     step_idx, m_policy_loss, m_value_loss, m_loss, speed)
            sum_train_data = 0.0
            sum_opt = 0.0
            writer.add_scalar("loss_policy", m_policy_loss, step_idx)
            writer.add_scalar("loss_value", m_value_loss, step_idx)
            writer.add_scalar("loss", m_loss, step_idx)
            writer.add_scalar("loss_policy_raw", m_policy_loss_raw, step_idx)
            writer.add_scalar("loss_value_raw", m_value_loss_raw, step_idx)
            writer.add_scalar("loss_raw", m_loss_raw, step_idx)
            writer.add_scalar("weights", m_weights, step_idx)
            writer.add_scalar("weights_min", m_weights_min, step_idx)
            writer.add_scalar("weights_max", m_weights_max, step_idx)
            writer.add_scalar("values", m_values, step_idx)
            writer.add_scalar("speed", speed, step_idx)

            if best_loss is None:
                best_loss = m_loss
            elif best_loss > m_loss:
                name = os.path.join(save_path, "best_%.4e.dat" % m_loss)
                torch.save(net.state_dict(), name)
                best_loss = m_loss

        if step_idx % config.push_scramble_buffer_iters == 0:
            if step_idx % config.push_scramble_buffer_iters * 3 and config.iterative_scramble_deepening:
                current_scramble_depth += 1
                log.info("Deepening scrambles, new depth = %d", current_scramble_depth)

            # scramble_buf.extend(model.make_scramble_buffer(cube_env, config.train_batch_size, current_scramble_depth))
            scramble_buf.extend(model.make_scramble_buffer(cube_env, config.train_batch_size * config.scramble_buffer_batches, current_scramble_depth))

            log.info("Pushed new data in scramble buffer, new size = %d", len(scramble_buf))

        if config.train_checkpoint_batches is not None and step_idx % config.train_checkpoint_batches == 0:
            name = os.path.join(save_path, "chpt_%06d.dat" % step_idx)
            torch.save(net.state_dict(), name)

        if config.train_max_batches is not None and config.train_max_batches <= step_idx:
            log.info("Limit of train batches reached, exiting")
            break

    writer.close()
