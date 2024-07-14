#%% 
import comet
import hydra 
from omegaconf import DictConfig, OmegaConf
import pyrootutils
from src import utils
import os
from pytorch_lightning import seed_everything
from src.utils import instantiator as inst
from src.utils.instantiator import load_object
from src.utils import setting_environment
from src.utils import task_wrapper

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": str(root / "configs"),
    "config_name": "train.yaml",
}
log = utils.get_pylogger(__name__)

print(__file__)


@task_wrapper
def train(cfg: DictConfig):
    #OmegaConf.resolve(cfg)

    log.info("Logging gpu info")
    utils.log_gpu_memory_metadata()

    if cfg.get("seed"):
        log.info("Setting seed")
        seed_everything(cfg.seed)
    
    log.info(f"Instantiating datamodule <<{cfg.datamodule._target_}>>")
    datamodule= hydra.utils.instantiate(cfg.datamodule,_recursive_=False)

    if cfg.visualize:
        datamodule.setup()
        log.info(f"Visualizing datamodule = <<{cfg.visualisation.verbose}>>") 
        datamodule_vis = hydra.utils.instantiate(cfg.visualisation,datamodule=datamodule)
        datamodule_vis.visualize()
    log.info(f'Instantiating callbacks:')
    callbacks = inst.instantiate_callbacks(cfg.callbacks)

    log.info(f'Instantiating loggers:')
    logger = inst.instantiate_loggers(cfg.logger)    

    log.info(f"Instantiating Lit - model <<{cfg.backbone._target_}>>")
    modelmodule = hydra.utils.instantiate(cfg.backbone)

    log.info(f"Instantiating Trainer <<{cfg.trainer._target_}>>")
    trainer = hydra.utils.instantiate(cfg.trainer,callbacks=callbacks,logger=logger) 

    log.info("Starting training")
    trainer.fit(model=modelmodule,datamodule=datamodule)
    best_model_path = trainer.checkpoint_callback.best_model_path

    log.info(f"Training complete- loading best model at :<{best_model_path}>")
    object_module = load_object(cfg.backbone._target_)
    modelmodule = object_module.load_from_checkpoint(best_model_path)

    log.info("Training complete - Starting testing")
    trainer.test(model=modelmodule,datamodule=datamodule)

    log.info('Logging model and its path')
    logger[0].experiment.log_parameters({"best_model_path":best_model_path})



    
    log.info(f"Instantiating Downstream task <<{cfg.downstream._target_}>>")
    dds = hydra.utils.instantiate(cfg.downstream,encoder=modelmodule.model.encoder)
    dds.fit(datamodule.train_dataloader())

    log.info(f"Instantiating evaluator <<{cfg.eval._target_}>>") 
    evaluator = hydra.utils.instantiate(cfg.eval,dds=dds,datamodule=datamodule,logger=logger)
    
    log.info("Starting benchmark")
    evaluator.setup()
    vas_score  = evaluator.evaluate_vas()
    sa_score = evaluator.evaluate_sa()

    return vas_score, sa_score
#%%
@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig):
    log.info("Starting training script")
    log.info("Setting environment variables")
    setting_environment(cfg.env)
    train(cfg)

#%%    
if __name__ == "__main__":
    main()