import hydra
from omegaconf import DictConfig
from pathlib import Path
from typing import Union
from src.train import run_study, view_study, train_w_best_params, test
from src.partitions import create_partitions
from src.inference import daily_prediction, range_prediction
from src.plots import optuna_plots, nn_confusion_calendar, first_confusion_calendar, confusion_calendar_patches
from src.explain import explain
from src.partitions import preprocessing
import pandas as pd


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    action = cfg.action
    if action == "create_partitions":
        create_partitions(cfg)
    elif action == "run_study": # if modified input features, we need to create partitions again before running this!
        run_study(cfg)
    elif action == "view_study":
        view_study(cfg)
    elif action == "train_w_best_params":
        train_w_best_params(cfg)
    elif action == "test":
        test(cfg)
    elif action == "daily_prediction":
        daily_prediction(cfg)
    elif action == "range_prediction":
        range_prediction(cfg)
    elif action == "explain":
        explain(cfg)
    elif action == "optuna_plots":
        optuna_plots(cfg)
    elif action == "preprocessing":
        preprocessing(cfg)
    elif action == "nn_confusion_calendar":
        nn_confusion_calendar(cfg)
    elif action == "first_confusion_calendar":
        first_confusion_calendar(cfg)
    elif action == "confusion_calendar_patches":
        confusion_calendar_patches()
    else:
        raise ValueError("Action not implemented.")

if __name__ == "__main__":
    main()