import torch
import wandb
import argparse
import numpy as np
import logging
import config
from config import logger
import pprint
from termcolor import colored
from collections import defaultdict
from utils import seed_everything, get_model, get_dataset, \
                  get_strategy, wandb_run_name

def get_default_parser():
    parser = argparse.ArgumentParser()
    for arg, default_value in config.DEFAULT_CONFIG.items():
        if type(default_value) is bool:
            parser.add_argument(f'--{arg}', action=f'store_{not default_value}'.lower())
        else:
            parser.add_argument(f'--{arg}', type=type(default_value), default=default_value)
    return parser

def main(args):
    # Possible Experiment:
    # - experiment the effect of different active learning strategies under different levels of noise
    # - experiment with malicious labelling and how different active learning strategies handle it
        # interesting - here it's important to define what sort of malicious labelling we're talking about
        # - define possible attacks (https://www.usenix.org/conference/usenixsecurity21/presentation/vicarte)
        # experiment with other types of accuracy: (https://arxiv.org/pdf/2107.01622.pdf)

    params = vars(args)

    # Training parameters
    params['optimizer_args'] = {'lr': args.lr}
    params['train_args'] = {'batch_size': args.batch_size, 'num_workers': args.num_workers}
    params['test_args'] = {'batch_size': args.batch_size, 'num_workers': args.num_workers}

    # Dataset parameters
    params['num_classes'] = 10
    if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
        params['image_channels'] = 1
        params['feature_map_dim'] = 16 * 4 * 4 * params['image_channels']
    if args.dataset == 'cifar10':
        params['image_channels'] = 3
        params['feature_map_dim'] = 16 * 5 * 5 * params['image_channels']

    # Initialize the dataset
    dataset = get_dataset(args.dataset, args.data_dir, args.num_valid, args.noise, args.noise_rate)

    # Initialize the model
    model = get_model(args.model, params)

    # Initialize the active learning strategy
    strategy = get_strategy(args.strategy, {'dataset': dataset, 'model': model})
      
    # Initialize cloud logging through wandb
    run_name = wandb_run_name(args)
    if args.wandb:
        wandb.init(project='active-learning', name=run_name,
                   config=params, reinit=True, entity='msc-ai',
                   tags=['experiments'])

    # Use the same seed for reproducibility
    seed_everything(args.seed)

    # start experiment
    dataset.initialize_labels(args.n_init_labeled)
    logger.debug(f"Initial number of labeled pool: {args.n_init_labeled}")
    logger.debug(f"Initial number of unlabeled pool: {dataset.n_pool-args.n_init_labeled}")
    logger.debug(f"Validation set: {dataset.n_valid}")
    logger.debug(f"Testing set: {dataset.n_test}")

    best_acc = .0

    # Initial round - train the model with the initial labeled pool
    logger.info("Starting Round 0")
    strategy.fit()
    preds = strategy.predict(dataset.get_validation_data())
    acc = (dataset.Y_valid == preds).sum().item() / len(dataset.Y_valid)
    logger.info(f"Round 0 validation accuracy: {colored(acc, 'green')}")

    # Active Learning Rounds
    for rd in range(1, args.n_round+1):
        logger.info(f"Starting Round {rd}")

        uniques = np.unique(dataset.Y_train[dataset.labeled_idxs], return_counts=True)
        pool_distribution = dict(zip(uniques[0], uniques[1]))
        pool_distribution_dict = "{\n"
        for key, value in pool_distribution.items():
            pool_distribution_dict += f"    {key}: {value},\n"
        pool_distribution_dict += "}"

        logger.debug("Labeled pool distribution:\n%s\n", pool_distribution_dict)

        # Query unlabeled pool
        query_idxs = strategy.query(args.n_query)

        logger.debug(f"Round {rd} queried digits: {dataset.Y_train[query_idxs]}\n")

        # Update the dataset with queried indices
        strategy.update(query_idxs)

        # Retrain the model with updated dataset
        strategy.fit()

        # Compute the test accuracy
        preds = strategy.predict(dataset.get_validation_data())
        acc = (dataset.Y_valid == preds).sum().item() / len(dataset.Y_valid)

        correct = np.unique(dataset.Y_valid[dataset.Y_valid == preds], return_counts=True)
        mistakes = np.unique(dataset.Y_valid[dataset.Y_valid != preds], return_counts=True)

        performance = defaultdict(lambda: [0, 0])

        # Loop through all digits (assuming digits range from 0 to 9)
        for digit in range(10):
            if digit in correct[0]:
                performance[digit][0] = correct[1][np.where(correct[0] == digit)][0]
            if digit in mistakes[0]:
                performance[digit][1] = mistakes[1][np.where(mistakes[0] == digit)][0]
                
        performance_str = "{\n"
        for key, (correct_count, mistake_count) in performance.items():
            performance_str += f"    {key}: [{colored(correct_count, 'green')}, {colored(mistake_count, 'red')}],\n"
        performance_str += "}"

        # Log the colored performance string
        logger.debug("Performance:\n%s\n", performance_str)
            
        # Save the best accuracy
        if acc > best_acc:
            best_acc = acc

        logger.info(f"Round {rd} validation accuracy: {colored(acc, 'green')}")
        logger.debug(f"Round {rd} labeled pool: {dataset.labeled_idxs.sum()}")
        logger.debug(f"Round {rd} unlabeled pool: {dataset.n_pool-dataset.labeled_idxs.sum()}")

        if args.wandb:
            wandb.log({'Validation Accuracy': acc, 'Round': rd})
            wandb.run.summary['Best Accuracy'] = best_acc
    
    # Final test error
    preds = strategy.predict(dataset.get_test_data())
    test_acc = (dataset.Y_test == preds).sum().item() / len(dataset.Y_test)
    test_error = 1 - test_acc
    logger.info(f"Final test accuracy: {test_acc}")
    logger.info(f"Final test error: {test_error}")

    if args.wandb:
        wandb.run.summary['Test Accuracy'] = test_acc
        wandb.run.summary['Test Error'] = test_error

if __name__ == '__main__':
    parser = get_default_parser()
    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.handlers[0].setLevel(logging.DEBUG)
        logger.debug('Debug mode on')

    if args.experiment:
        for seed in np.random.randint(0, 1000, 3):
            logger.info(f"Running experiment with seed {seed}")
            args.seed = seed
            main(args)
    else:
        logger.info(f"Running experiment with seed {args.seed}")
        main(args)
    logger.info(f"Finished experiment for configuration:\n{vars(args)}")
