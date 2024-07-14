#!/usr/bin/env python3
import argparse
import coloredlogs
import logging

from a3c import run_a3c
from time import sleep

logger = logging.getLogger()


def main():
    fmt = "%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s"
    coloredlogs.install(level='DEBUG', fmt=fmt)
    logger.setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser()

    # general setups
    parser.add_argument('--folder', type=str, default=None, help='name the result folder')
    parser.add_argument('--append-experiment-num', type=str, default=None,
                        help='experiment identifier (date)')
    parser.add_argument('--parallel-size', type=int, default=16,
                        help='parallel thread size')
    parser.add_argument('--gym-env', type=str, default='MsPacmanNoFrameskip-v4',
                        help='OpenAi Gym environment ID')
    parser.add_argument('--save-to', type=str, default='results/',
                        help='the directory to save results')
    parser.add_argument('--log-interval', type=int, default=500, help='logging info frequency')
    parser.add_argument('--performance-log-interval', type=int, default=1000, help='logging info frequency')

    # checkpoint, e.g. --checkpoint-freq=5 saves the agent every 5 million steps
    parser.add_argument('--checkpoint-freq', type=int, default=1,
                        help='checkpoint frequency, default to every eval-freq*checkpoint-freq steps')

    # setup network architecture
    parser.add_argument('--use-mnih-2015', action='store_true',
                        help='use Mnih et al [2015] network architecture, if a3c will add value output layer')
    parser.set_defaults(use_mnih_2015=False)
    parser.add_argument('--input-shape', type=int, default=84,
                        help='84x84 as default')
    parser.add_argument('--padding', type=str, default='VALID',
                        help='VALID or SAME')

    # setup A3C components
    parser.add_argument('--local-t-max', type=int, default=20,
                        help='repeat step size')
    parser.add_argument('--rmsp-alpha', type=float, default=0.99,
                        help='decay parameter for RMSProp')
    parser.add_argument('--rmsp-epsilon', type=float, default=1e-5,
                        help='epsilon parameter for RMSProp')
    parser.add_argument('--initial-learn-rate', type=float, default=7e-4,
                        help='initial learning rate for RMSProp')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards')
    parser.add_argument('--entropy-beta', type=float, default=0.01,
                        help='entropy regularization constant')
    parser.add_argument('--max-time-step', type=float, default=10 * 10**7,
                        help='maximum time step, use to anneal learning rate')
    parser.add_argument('--max-time-step-fraction', type=float, default=1.,
                        help='ovverides maximum time step by a fraction')
    parser.add_argument('--grad-norm-clip', type=float, default=0.5,
                        help='gradient norm clipping')
    parser.add_argument('--eval-freq', type=int, default=1000000,
                        help='how often to evaluate the agent')
    parser.add_argument('--eval-max-steps', type=int, default=125000,
                        help='number of steps for evaluation')
    parser.add_argument('--l2-beta', type=float, default=0.,
                        help='L2 regularization beta')

    # Alternatives to reward clipping
    parser.add_argument('--unclipped-reward', action='store_true',
                        help='use raw reward')
    parser.set_defaults(unclipped_reward=False)
    # Ape-X Pohlen, et. al 2018, this is the TB operator
    parser.add_argument('--transformed-bellman', action='store_true',
                        help='use transformed bellman equation')
    parser.set_defaults(transformed_bellman=False)
    parser.add_argument('--reward-constant', type=float, default=2.0,
                        help='value added to all non-zero rewards when using'
                             ' transformed bellman operator')

    args = parser.parse_args()

    if args.unclipped_reward:
        assert args.transformed_bellman # must use TB if raw rewards
        logger.info('Running A3CTB...')
    else:
        logger.info('Running A3C...')

    sleep(2)
    run_a3c(args)


if __name__ == "__main__":
    main()
