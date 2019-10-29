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
    writer = SummaryWriter(comment="-" + name)
    save_path = os.path.join("saves", name)
    os.makedirs(save_path)

    cube_env = cubes.get(config.cube_type)
    assert isinstance(cube_env, cubes.CubeEnv)
    log.info("Selected cube: %s", cube_env)
    value_targets_method = model.ValueTargetsMethod(config.train_value_targets_method)

    net = model.Net(cube_env.encoded_shape, len(cube_env.action_enum)).to(device)
    print(net)
    opt = optim.Adam(net.parameters(), lr=config.train_learning_rate)
    sched = scheduler.StepLR(opt, 1, gamma=config.train_lr_decay_gamma) if config.train_lr_decay_enabled else None

    step_idx = 0
    buf_policy_loss, buf_value_loss, buf_loss = [], [], []
    buf_policy_loss_raw, buf_value_loss_raw, buf_loss_raw = [], [], []
    buf_mean_values = []
    ts = time.time()
    best_loss = None

    log.info("Generate scramble buffer...")
    scramble_buf = collections.deque(maxlen=config.scramble_buffer_batches*config.train_batch_size)
    scramble_buf.extend(model.make_scramble_buffer(cube_env, config.train_batch_size*2, config.train_scramble_depth))
    log.info("Generated buffer of size %d", len(scramble_buf))

    while True:
        if config.train_lr_decay_enabled and step_idx % config.train_lr_decay_batches == 0:
            sched.step()
            log.info("LR decrease to %s", sched.get_lr()[0])
            writer.add_scalar("lr", sched.get_lr()[0], step_idx)

        step_idx += 1
        x_t, weights_t, y_policy_t, y_value_t = model.sample_batch(
            scramble_buf, net, device, config.train_batch_size, value_targets_method)

        opt.zero_grad()
        policy_out_t, value_out_t = net(x_t)
        value_out_t = value_out_t.squeeze(-1)
        value_loss_t = (value_out_t - y_value_t)**2
        value_loss_raw_t = value_loss_t.mean()
        if config.weight_samples:
            value_loss_t *= weights_t
        value_loss_t = value_loss_t.mean()
        policy_loss_t = F.cross_entropy(policy_out_t, y_policy_t, reduction='none')
        policy_loss_raw_t = policy_loss_t.mean()
        if config.weight_samples:
            policy_loss_t *= weights_t
        policy_loss_t = policy_loss_t.mean()
        loss_raw_t = policy_loss_raw_t + value_loss_raw_t
        loss_t = value_loss_t + policy_loss_t
        loss_t.backward()
        opt.step()

        # save data
        buf_mean_values.append(value_out_t.mean().item())
        buf_policy_loss.append(policy_loss_t.item())
        buf_value_loss.append(value_loss_t.item())
        buf_loss.append(loss_t.item())
        buf_loss_raw.append(loss_raw_t.item())
        buf_value_loss_raw.append(value_loss_raw_t.item())
        buf_policy_loss_raw.append(policy_loss_raw_t.item())

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
            writer.add_scalar("values", m_values, step_idx)
            writer.add_scalar("speed", speed, step_idx)

            if best_loss is None:
                best_loss = m_loss
            elif best_loss > m_loss:
                name = os.path.join(save_path, "best_%.4e.dat" % m_loss)
                torch.save(net.state_dict(), name)
                best_loss = m_loss

        if step_idx % config.push_scramble_buffer_iters == 0:
            scramble_buf.extend(model.make_scramble_buffer(cube_env, config.train_batch_size,
                                                           config.train_scramble_depth))
            log.info("Pushed new data in scramble buffer, new size = %d", len(scramble_buf))

        if config.train_checkpoint_batches is not None and step_idx % config.train_checkpoint_batches == 0:
            name = os.path.join(save_path, "chpt_%06d.dat" % step_idx)
            torch.save(net.state_dict(), name)

        if config.train_max_batches is not None and config.train_max_batches <= step_idx:
            log.info("Limit of train batches reached, exiting")
            break

    writer.close()
