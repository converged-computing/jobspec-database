from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None, config_path='./config/', config_name='train.yaml')
def my_app(cfg: DictConfig) -> None:
    #print(OmegaConf.to_yaml(cfg))
    from train import train
    from util import extras
    extras(cfg)

    return train(cfg)

if __name__ == "__main__":
    my_app()