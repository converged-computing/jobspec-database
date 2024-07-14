import argparse
import sys


USAGE_TEXT = '''python experiment_topology.py <command> [<args>]

The available commands are:
   inspect             Inspects datasets that will be used for training and validation
   train               Train new model for the experiment
   evaluate            Evaluate the trained model against annotated muscima parts
   evaluate_on_primus  Evaluate trained model against some primus incipits
   evaluate_on_real    Evaluate trained model against real scanned sheet music

Experiment Topology:
    - Train on 31.5K Primus incipits and 31.5K generated incipits
    - Validate on 1K Primus incipits
    - Generates images with staves above and below
    - Use symbols from all writers except for the evaluating ones
    - Minimal staff width of 1200px

    USE ONLY SYNTHETIC SYMBOLS FROM GIVEN FOLDER at "--symbols"

'''


class ExperimentSymbols(object):
    def __init__(self):
        parser = argparse.ArgumentParser(usage=USAGE_TEXT)
        parser.add_argument('command', help='Command to run')
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def _prepare_datasets(self):
        from mashcima import Mashcima
        mc = Mashcima(
            use_cache=True,
            skip_writers=[13, 17, 20, 34, 41, 49],
            muscima_fraction=0.0, # muscima is already mixed into synthetic symbols
            generated_fraction=1.0 # take only synthetic symbols
        )

        from experiment_utils import prepare_annotations
        training_annotations = prepare_annotations(
            primus_skip=0, primus_take=31564, generated_take=31564
        )
        validation_annotations = prepare_annotations(
            primus_skip=63127, primus_take=1000, generated_take=0
        )

        from experiment_utils import prepare_dataset
        training_dataset = prepare_dataset(
            mc, training_annotations, min_staff_with=1200, single_staff=False
        )
        validation_dataset = prepare_dataset(
            mc, validation_annotations, min_staff_with=1200, single_staff=False
        )

        return training_dataset, validation_dataset

    def inspect(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--quiet', action="store_true", help="store files insted of popping a window")
        parser.add_argument('--symbols', default="datasets/topology", type=str)
        args = parser.parse_args(sys.argv[2:])

        # set symbols repository
        import config
        config.SYNTHETIC_SYMBOLS_PATH = args.symbols

        directory = "tf-logs/visual-check/" + args.symbols.split("/")[-1]

        training_dataset, validation_dataset = self._prepare_datasets()

        print("\n\nInspecting TRAINING dataset: (20 items)")
        training_dataset.check_dataset_visually(
            example_count=20,
            quiet=args.quiet,
            directory=directory + "/training"
        )

        print("\n\nInspecting VALIDATION dataset: (20 items)")
        validation_dataset.check_dataset_visually(
            example_count=20,
            quiet=args.quiet,
            directory=directory + "/validation"
        )

    def train(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', default="experiment_topology")
        parser.add_argument('--epochs', default=20, type=int)
        parser.add_argument('--batch_size', default=10, type=int)
        parser.add_argument('--threads', default=4, type=int)
        parser.add_argument('--load_model', action="store_true", help="continue training a model")
        parser.add_argument('--seed_offset', default=0, type=int)
        parser.add_argument('--symbols', default="datasets/topology", type=str)
        args = parser.parse_args(sys.argv[2:])

        # set symbols repository
        import config
        config.SYNTHETIC_SYMBOLS_PATH = args.symbols

        # set seed
        import tensorflow as tf
        import numpy as np
        import random
        tf.set_random_seed(20200524 + args.seed_offset)
        np.random.seed(20200524 + args.seed_offset)
        random.seed(20200524 + args.seed_offset)

        training_dataset, validation_dataset = self._prepare_datasets()

        # train
        from app.Network import Network

        # continue training
        if args.load_model:
            # load old one
            print("Loading old model...")
            network = Network.load(args.model)
        else:
            # delete old one
            if Network.exists(args.model):
                if input("Type 'yes' to delete the old, trained model.") != "yes":
                    exit("No model will be overwritten")
                print("Deleting old model...")
                Network.delete_model(args.model)

            # create new one
            print("Creating new model...")
            network = Network(
                name=args.model,
                continual_saving=True,
                create_logdir=True,
                threads=args.threads
            )

        network.train(
            training_dataset,
            validation_dataset,
            epochs=args.epochs,
            batch_size=args.batch_size
        )

    def evaluate(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', default="experiment_topology")
        parser.add_argument('--writers', type=str, help="writers to evaluate on e.g. '1,2,3'")
        parser.add_argument('--pages', type=str, help="pages to evaluate on e.g. '1,2,3'")
        args = parser.parse_args(sys.argv[2:])

        from experiment_evaluation import evaluate_model
        evaluate_model(args.model, args.writers, args.pages)
    
    def evaluate_on_real(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', default="experiment_topology")
        args = parser.parse_args(sys.argv[2:])

        from experiment_evaluation import evaluate_on_real
        evaluate_on_real(args.model)

    def evaluate_on_primus(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', default="experiment_topology")
        args = parser.parse_args(sys.argv[2:])

        from experiment_evaluation import evaluate_on_primus
        evaluate_on_primus(args.model)
    
    def aggregate_evaluation(self):
        """Goes throught the finished evaluations and prints their aggregates"""
        import os
        import numpy as np

        parser = argparse.ArgumentParser()
        parser.add_argument('--stdouts', default="tf-logs/stdout-eval")
        args = parser.parse_args(sys.argv[2:])

        directory = args.stdouts

        def get_ser_avg_from_file(f):
            in_averages = False
            for line in f:
                if line.startswith("=                Averages                ="):
                    in_averages = True
                if in_averages:
                    if line.startswith("avg SER:"):
                        return float(line.split(" ")[2])
            return None
        
        for dataset in ["muscima", "primus", "cavatina"]:
            print()
            print("====================")
            print("DATASET " + dataset)
            print("====================")
            print()

            SERs = {
                "L2": [], "L4": [], "L5": [], "L6": [],
                "L8": [], "L10": [],
                
                "A": [], "B": [], "C": [], "D": [],
                "E": [], "F": [], "G": [], "H": [],
            }
            for filename in os.listdir(directory):
                for experiment in SERs.keys():
                    if filename.startswith(experiment) and filename.endswith(dataset + ".txt"):
                        with open(os.path.join(directory, filename), "r") as f:
                            s = get_ser_avg_from_file(f)
                            if s is not None:
                                SERs[experiment].append(s)
            print(SERs)
            print()

            def fmt(key):
                values = SERs[key]
                if len(values) == 0:
                    return key + ": ?              "
                stddev = np.array(values).std()
                avg = np.array(values).mean()
                return key + f": {avg:.4f} +-{stddev:.4f}"

            print(dataset + ":")
            print(fmt("A"), fmt("B"), fmt("C"), fmt("D"), sep=" | ")
            print(fmt("E"), fmt("F"), fmt("G"), fmt("H"), sep=" | ")
            print()

            print(dataset + ":")
            dims = [2, 4, 5, 6, 8, 10]
            for d in dims:
                print(fmt("L" + str(d)))
            print()



if __name__ == '__main__':
    ExperimentSymbols()
