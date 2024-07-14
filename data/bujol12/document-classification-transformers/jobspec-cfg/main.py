import argparse
import logging

from src.experiment import Experiment
from src.metrics import Metrics
from src.json_document_dataset import JsonDocumentDataset

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Run a Document-classification model')

    parser.add_argument('--config', dest='config', type=str, default=None, required=True,
                        help='Path to the JSON config file')
    parser.add_argument('--train', dest='train_dataset', type=str, default=None,
                        help='Path to the training dataset')
    parser.add_argument('--eval', dest='eval_dataset', type=str, default=None,
                        help='Path to the eval dataset')
    parser.add_argument('--test', dest='test_dataset', type=str, default=None,
                        help='Path to the test dataset')
    parser.add_argument('--random', dest='random', type=bool, default=False,
                        help='If set to true, only run random baseline')

    return parser.parse_args()

K_THRES = 0.10

if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    experiment = Experiment(config_filepath=args.config, train_data_filepath=args.train_dataset,
                            eval_data_filepath=args.eval_dataset)

    if args.random:
        # just do random baseline
        if args.test_dataset is not None:

            test_dataset = JsonDocumentDataset(args.test_dataset, experiment.config)

            test_dataset.set_random_baseline(K_THRES)
            results = Metrics(test_dataset, 0, K_THRES)

            print()
            logger.info("----------Final Random Test Performance---------")
            logger.info(results.to_json())

            exit(0)

    experiment.train()

    results = experiment.eval(experiment.eval_dataset)

    print()
    logger.info("----------Final Eval Performance---------")
    logger.info(results.to_json())

    logger.info(f"Saving the model & results to: {experiment.experiment_folder}")
    experiment.save_results(results, "eval_results")
    experiment.save_model("final_model.torch")

    experiment.save_predictions(experiment.eval_dataset, "eval_predictions.json")

    if args.test_dataset is not None:
        logger.info("Running test dataset...")

        test_dataset = JsonDocumentDataset(args.test_dataset, experiment.config)

        results = experiment.eval(test_dataset)

        print()
        logger.info("----------Final Test Performance---------")
        logger.info(results.to_json())

        experiment.save_results(results, "test_results")
        experiment.save_predictions(test_dataset, "test_predictions.json")

