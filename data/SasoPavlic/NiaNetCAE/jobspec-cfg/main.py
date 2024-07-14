import argparse
import uuid
from datetime import datetime
from pathlib import Path

import torch
import yaml
from lightning.pytorch import seed_everything

import nianetcae
from log import Log
from nianetcae.cae_run import solve_architecture_problem
from nianetcae.dataloaders.nyu_dataloader import NYUDataset
from nianetcae.storage.database import SQLiteConnector


if __name__ == '__main__':

    RUN_UUID = uuid.uuid4().hex
    torch.set_float32_matmul_precision("medium")
    parser = argparse.ArgumentParser(description='Generic runner for Convolutional AE models')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/main_config.yaml')

    parser.add_argument('--algorithms', '-alg',
                        dest="algorithms",
                        metavar='list_of_strings',
                        help='NIA algorithms to use')

    args = parser.parse_args()

    with open(args.filename, 'r') as file:
        try:
            config = yaml.load(file, Loader=yaml.Loader)  # yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print("Error while loading config file")
            print(exc)

    config['logging_params']['save_dir'] += RUN_UUID + '/'
    Path(config['logging_params']['save_dir']).mkdir(parents=True, exist_ok=True)

    Log.enable(config['logging_params'])
    Log.info(f'Program start: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
    Log.info(f"RUN UUID: {RUN_UUID}")
    Log.header("NiaNetCAE settings")
    Log.info(config)

    conn = SQLiteConnector(config['logging_params']['db_storage'], f"solutions")  # _{RUN_UUID}")
    seed_everything(config['exp_params']['manual_seed'], True)

    datamodule = NYUDataset(**config["data_params"])
    datamodule.setup()

    nianetcae.cae_run.RUN_UUID = RUN_UUID
    nianetcae.cae_run.config = config
    nianetcae.cae_run.conn = conn
    nianetcae.cae_run.datamodule = datamodule

    algorithms = []
    if args.algorithms is not None:
        args.algorithms = args.algorithms.split(',')
        algorithms = args.algorithms
    else:
        algorithms = config['nia_search']['algorithms']

    solve_architecture_problem(algorithms)
    Log.info(f'\n Program end: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
