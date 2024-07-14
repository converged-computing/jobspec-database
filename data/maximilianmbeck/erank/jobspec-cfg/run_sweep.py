from omegaconf import DictConfig
from ml_utilities.utils import get_config_file_from_cli, get_config
from pathlib import Path

from ml_utilities.runner import run_sweep

def run(cfg: DictConfig):
    run_sweep(cfg)

if __name__=='__main__':
    cfg_file = get_config_file_from_cli(config_folder='configs', script_file=Path(__file__))
    cfg = get_config(cfg_file)
    run(cfg)