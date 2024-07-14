"""
File to train a model for image classificaton
https://github.com/ashleve/lightning-hydra-template/tree/e27c11c34254f435cfb5c8f6ab7b7f747020621a
"""
import hydra
from omegaconf import DictConfig

from dataclasses import dataclass
from typing import Any

# @hydra.main()
@hydra.main(config_path="config")
def main(config: DictConfig) -> None:
    from baseline.train_optuner.train_v3 import train
    from baseline.utils import template_utils

    template_utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        template_utils.print_config(config, resolve=True)

    # Train model
    return train(config)


########
# Main #
########
if __name__ == "__main__":
    main()