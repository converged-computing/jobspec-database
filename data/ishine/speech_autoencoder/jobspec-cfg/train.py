import torch
import argparse
from dataset import LibriSpeech
import importlib
from transformers import AutoConfig
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor,StochasticWeightAveraging
from utils.utils import *
# importlib.reload(SAE)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configs.SpeechWGAN', help='config path')
    parser.add_argument('--mode', type=str,
                        default='test', help='train or test?')
    parser.add_argument('--checkpoint', type=str,
                        default=None, help='checkpoint path')
    return parser.parse_args()

if __name__ == '__main__':
    os.environ['HF_DATASETS_OFFLINE']="1"
    os.environ['TRANSFORMERS_OFFLINE']="1"
    args = get_args()
    config = importlib.import_module(args.config)
    task_config = config.task_config
    train_config = config.train_config
    dataset_config = config.dataset_config

    # device = torch.device('cuda:'+str(train_config['gpus'])
    #                       if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    seed_everything(42)
    # torch.use_deterministic_algorithms(False)

    check_mk_dirs(train_config['log_path'])
    data = LibriSpeech(**dataset_config)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=train_config['log_path'],
        filename='{epoch:02d}-{val_loss:.6f}',
        save_top_k=10,
        save_last=True,
        mode='min')
    callbacks = [checkpoint_callback]
    tb_logger = pl_loggers.TensorBoardLogger(train_config['log_path'])

    if args.mode == 'train':
        model = config.task(**task_config) #if '1' in train_config['stage'] else Model_Stage_2(
            # model_config, train_config)
        # print(model)
        trainer = Trainer(
            accelerator="gpu", 
            strategy="ddp",
            devices=train_config["devices"],
            max_epochs=train_config["max_epochs"],
            callbacks=callbacks,
            default_root_dir=train_config['log_path'],
            deterministic=train_config['deterministic'],
            logger=tb_logger,
            num_sanity_val_steps=1,
            # precision=32
            # num_sanity_val_steps=1, #if dataset_config['stage'] == '2' else 2,
            # log_every_n_steps=40,
            # check_val_every_n_epoch=50,
        )
        trainer.fit(
            model=model,
            datamodule=data)

    else:
        model = config.task(**task_config) #if '1' in train_config['stage'] else Model_Stage_2(
        # state_dict = torch.load(args.checkpoint,map_location=device)['state_dict']
        # model.load_state_dict(state_dict)
        print(model)
        trainer = Trainer(
            accelerator="gpu", 
            strategy="dp",
            devices=train_config["devices"],
            max_epochs=train_config["max_epochs"],
            callbacks=callbacks,
            default_root_dir=train_config['log_path'],
            deterministic=train_config['deterministic'],
            logger=tb_logger,
            # num_sanity_val_steps=1, #if dataset_config['stage'] == '2' else 2,
            # log_every_n_steps=40,
            # check_val_every_n_epoch=50,
            resume_from_checkpoint=args.checkpoint)

        if args.mode == 'continue':
            trainer.fit(model=model, datamodule=data)

        elif args.mode == 'test':
            trainer.test(model=model, datamodule=data)

        elif args.mode == 'predict':
            trainer.predict(model=model, datamodule=data)
