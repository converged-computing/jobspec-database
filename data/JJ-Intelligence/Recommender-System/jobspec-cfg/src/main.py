import argparse
import os
import time
from memory_profiler import memory_usage

import numpy as np
import pandas as pd

from evaluation import to_cross_validation_datasets
from models import MatrixFactoriser, RandomModel, KNNBenchmark, KNNModel
from io_handler import read_train_csv, read_test_csv, write_output_csv
from models.industry_benchmark import SVDBenchmark
from train import start_training


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('run_option', choices=['run', 'tune', 'load', 'evaluate'])
    parser.add_argument('--trainfile', type=str, help='File containing the train data')
    parser.add_argument('--testfile', type=str, help='File containing the test data')
    parser.add_argument('--outputfile', type=str, help='File to output predictions')
    parser.add_argument('--checkpointfile', type=str, help='Checkpoint file to load')
    parser.add_argument('--evalmodel', type=int, help='Model to evaluate')
    parser.add_argument('--model', type=str,
                        help='Model to use when \'run_option\' is \'run\' (MatrixFact/Random/Baseline)', default=None)
    args = parser.parse_args()

    if args.run_option == "tune":

        print("Reading training CSV")
        train_dataset, evaluation_dataset, test_dataset = read_train_csv(args.trainfile, test_size=0.1, eval_size=0.1)

        print("\n---- Starting training ----")
        an = start_training(train_dataset, evaluation_dataset)
        print("\n---- Finished training ----")

        print("\nBest trial:")
        print(an.best_trial)
        print("\nBest checkpoint:")
        print(an.best_checkpoint)
        print("\nBest config:")
        print(an.best_config)
        print("\nBest result:")
        print(an.best_result)

        print("\n---- Loading best checkpoint model ----")
        model = MatrixFactoriser()
        model.load(os.path.join(an.best_checkpoint, "checkpoint.npz"))

        print("MSE on test dataset")
        print(model.eval(test_dataset))

        print("Reading prediction dataset")
        predict_dataset = read_test_csv(args.testfile)

        print("Creating predictions")
        predictions = model.predict(predict_dataset)

        print("Writing prediction output")
        write_output_csv(args.outputfile, predict_dataset, predictions)

    elif args.run_option == "run":
        if args.model is None:
            # Default to matrix fact
            model_name = "matrixfact"
        else:
            model_name = args.model.lower()

        print("Reading training CSV")
        train_dataset, test_dataset = read_train_csv(args.trainfile, test_size=0.1, eval_size=0)

        print("Starting training")
        if model_name == 'matrixfact':
            model = MatrixFactoriser()
            model.initialise(k=32, hw_init_stddev=0.014676120289293371)
            model.train(
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                epochs=70,
                batch_size=16_384,
                lr=0.0068726720195871754,
                user_reg=0.0676216799448991,
                item_reg=0.06639363622316222,
                user_bias_reg=0.12389941928866091,
                item_bias_reg=0.046243201501061273,
            )

        # model.save("model.npz")

        elif model_name == 'average':
            model = RandomModel()
            model.initialise(is_normal=False)

            print("Starting training")
            model.train(
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
            )

        elif model_name == 'random':
            model = RandomModel()
            model.initialise(is_normal=True)

            print("Starting training")
            model.train(
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
            )

        elif model_name == 'knn':
            print("Training baseline model")
            model = KNNBenchmark()
            model.initialise(knn_class="KNNBasic")
            model.train(train_dataset)
        elif model_name == 'custom knn':
            print("Training KNN model")
            model = KNNModel()
            model.train(train_dataset)
        elif model_name == 'svd':
            print("Training baseline model")
            model = SVDBenchmark()
            model.initialise()
            model.train(train_dataset)
        else:
            raise RuntimeError("Invalid argument for 'run_model'")

        print("Run on test set")
        evaluation = model.eval(test_dataset)
        print(evaluation)

        print("Reading prediction dataset")
        predict_dataset = read_test_csv(args.testfile)

        print("Creating predictions")
        predictions = model.predict(predict_dataset)

        print("Writing prediction output")
        write_output_csv(args.outputfile, predict_dataset, predictions)

    elif args.run_option == "load":
        print("Loading model from:", args.checkpointfile)
        model = MatrixFactoriser()
        model.load(args.checkpointfile)

        print("Reading prediction dataset")
        predict_dataset = read_test_csv(args.testfile)

        print("Reading training CSV to test MSE")
        train_dataset, test_dataset = read_train_csv(args.trainfile, test_size=1)
        ev = model.eval(test_dataset)
        print("MSE on training set:", ev.mse)

        print("Creating predictions")
        predictions = model.predict(predict_dataset)

        print("Writing prediction output")
        write_output_csv(args.outputfile, predict_dataset, predictions)

    elif args.run_option == "evaluate":
        # List of (model class, name, init kwargs, train kwargs)
        models = [
            (
                # Random model with a normal distribution
                RandomModel, "RandomNormal Baseline", {"is_normal": True}, {},
            ),
            (
                # Random model with a normal distribution
                RandomModel, "GlobalAverage Baseline", {"is_normal": False}, {},
            ),
            (
                KNNBenchmark, "KnnBasic Baseline", {"knn_class": "KNNBasic",  "k": 40}, {}
            ),
            (
                KNNBenchmark, "KnnBaseline Baseline", {"knn_class": "KNNBaseline", "k": 40}, {}
            ),
            (
                SVDBenchmark, "SVD Baseline", {}, {}
            ),
            (
                # Our model
                MatrixFactoriser,
                "SVD++",
                {"k": 32, "hw_init_stddev": 0.014676120289293371},
                {"epochs": 70, "batch_size": 16_384, "lr": 0.0068726720195871754, "user_reg": 0.0676216799448991,
                 "item_reg": 0.06639363622316222, "user_bias_reg": 0.12389941928866091,
                 "item_bias_reg": 0.046243201501061273},
            )
        ]

        print("Loading dataset")
        train_dataset = read_train_csv(args.trainfile, test_size=0., eval_size=0.)

        results = []

        chosen_models = models if args.evalmodel is None else [models[args.evalmodel]]

        for cv_num, (train_dataset, test_dataset) in enumerate(
                to_cross_validation_datasets(train_dataset, n_splits=5, seed=1)):

            for model_cls, name, init_kwargs, train_kwargs in chosen_models:
                # Run model on CV fold
                print("Evaluating '%s' on CV fold %d" % (name, cv_num))
                model = model_cls()
                model.initialise(**init_kwargs)
                start_time = time.time()
                mem_usage = memory_usage(lambda: model.train(train_dataset, **train_kwargs))
                end_time = time.time()
                evaluation = model.eval(test_dataset, train_time=end_time - start_time, max_mem_usage=max(mem_usage))
                print("> Results:\n", evaluation)

                # Update results
                results.append([cv_num, name, *evaluation.__dict__.values()])

        # Average model results
        for _, name, _, _ in chosen_models:
            model_results = [r[2:] for r in results if r[1] == name]
            model_mean = np.mean(model_results, axis=0)
            results.append(["Average", name, *model_mean])

        # Output a CSV
        cv_df = pd.DataFrame(results, columns=["CV Fold", "Model", *evaluation.__dict__.keys()])
        print("Final evaluation results:\n", cv_df.to_markdown(), sep="")
        cv_df.to_csv(args.outputfile)


if __name__ == "__main__":
    main()
