import pandas as pd
import argparse


def _parse_args():
    parser = argparse.ArgumentParser(description="Launch a job from the grid")
    parser.add_argument(
        "--num",
        "-n",
        default="1",
        help="Number of the job to launch",
    )
    parser.add_argument(
        "--exp",
        "-e",
        help="Number of the experiment to launch (overrides --num in the result)",
    )
    parser.add_argument(
        "--perm",
        "-p",
        action="store_true",
        help="Launch a job for permutation learning",
    )
    parser.add_argument(
        "--ds",
        "-d",
        default="mnli",
        help="Dataset to use (only for permutation learning)",
    )
    return parser.parse_args()


args = _parse_args()


if args.perm:
    hyper_params_all = pd.read_csv(
    "sh_scripts/python_helpers/moebert_perm_distil_grid.csv",
    sep=";",
    index_col=0,
    dtype={
        "experiment_number": int,
        "batch_size": int,
        "learning_rate": float,
        "weight_decay": float,
        "entropy": float,
        "gamma": float,
        "distill": float,
        "dataset": str,
        "perm_epoch": float,
    },
    )

    hyper_params = hyper_params_all[hyper_params_all.dataset == args.ds]
    hyper_params.drop(columns=["dataset", "experiment_number"], inplace=True)

    # set experiment_num as index
    hyper_params.index = range(1, 101)

    
    jobs_params = hyper_params.loc[int(args.num)]
    
    if args.exp is not None:
        print(
            f"{int(args.exp)} {int(jobs_params.batch_size)} {jobs_params.weight_decay} {jobs_params.learning_rate} {jobs_params.entropy} {jobs_params.gamma} {jobs_params.distill} {jobs_params.perm_epoch}"
        )
    else:
        print(
            f"{jobs_params.name} {int(jobs_params.batch_size)} {jobs_params.weight_decay} {jobs_params.learning_rate} {jobs_params.entropy} {jobs_params.gamma} {jobs_params.distill} {jobs_params.perm_epoch}"
        )
    
else:
    hyper_params = pd.read_csv(
    "sh_scripts/python_helpers/moebert_distil_grid.csv",
    sep=";",
    index_col=0,
    dtype={
        "experiment_name": int,
        "batch_size": int,
        "learning_rate": float,
        "weight_decay": float,
        "entropy": float,
        "gamma": float,
        "distill": float,
    },
    )
    jobs_params = hyper_params.loc[int(args.num)]
    if args.exp is not None:
        print(
            f"{int(args.exp)} {int(jobs_params.batch_size)} {jobs_params.weight_decay} {jobs_params.learning_rate} {jobs_params.entropy} {jobs_params.gamma} {jobs_params.distill}"
        )
    else:
        print(
            f"{jobs_params.name} {int(jobs_params.batch_size)} {jobs_params.weight_decay} {jobs_params.learning_rate} {jobs_params.entropy} {jobs_params.gamma} {jobs_params.distill}"
        )
