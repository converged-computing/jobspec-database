#!/bin/bash

#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=03:59:00
#SBATCH --output=./slurm_out/mdcrl-%j.out
#SBATCH --error=./slurm_err/mdcrl-%j.err

module load miniconda/3
conda activate mdcrl

export WANDB_API_KEY=1406ef3255ef2806f2ecc925a5e845e7164b5eef
wandb login

export LD_PRELOAD=/home/mila/s/sayed.mansouri-tehrani/MD-CRL/hack.so
# export WANDB_MODE=offline



# for runs more than a day, use: 1-11:59:00 (day-hour)


# -------------------------- Synthetic Mixing -------------------------- #

# python run_training.py ckpt_path=null model.optimizer.lr=0.01 model=mixing_synthetic datamodule=mixing ~callbacks.visualization_callback model.penalty_weight=0.00001
# python3 run_training.py model.additional_logger.logging_interval=400 ~callbacks.visualization_callback ~callbacks.early_stopping callbacks.model_checkpoint.monitor="train_loss" logger.wandb.tags=["mila"] ckpt_path=null trainer.max_epochs=2000


# python run_training.py ckpt_path=null model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist model=default model.autoencoder.num_channels=3 model.z_dim=32
# python run_training.py ckpt_path=null model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist model=multi_domain_autoencoder model.autoencoder.num_channels=3 model.z_dim=32
# python run_training.py ckpt_path=null model.optimizer.lr=0.001 datamodule/dataset=multi_domain_mnist model=default model.autoencoder.num_channels=3

# python run_training.py ckpt_path=null trainer.accelerator='gpu' trainer.devices=1 model/optimizer=adam model.optimizer.lr=0.001 model/scheduler_config=reduce_on_plateau model=mixing_synthetic model.penalty_criterion="minmax" model.hinge_loss_weight=0.0 datamodule=mixing datamodule.dataset.num_domains=4 datamodule.dataset.z_dim=8 ~callbacks.visualization_callback model.penalty_weight=1.0 logger.wandb.tags=["mila","test"]

# ------------------------------------------------------------------------------------- #
# ----------------------- Polynomial Mixing, Non-Linear Model ------------------------- #
# ------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------- #
# ----------------------------------- Just reconstruction ----------------------------- #

# python run_training.py ckpt_path=null model=mixing_synthetic model/autoencoder=poly_ae model.optimizer.lr=0.001 datamodule=mixing datamodule.dataset.linear=False datamodule.dataset.non_linearity=polynomial datamodule.dataset.polynomial_degree=3 datamodule.batch_size=64 datamodule.dataset.z_dim=8 model.z_dim=8 datamodule.dataset.num_domains=8 datamodule.dataset.x_dim=200 ~callbacks.visualization_callback logger.wandb.tags=["mila","poly-mixing"]

# ------------------------------------------------------------------------------------- #
# ------------------------------------- Disentanglement ------------------------------- #

# p=2, d=6
# python run_training.py ckpt_path=null model=mixing_md_encoded_autoencoder model.optimizer.lr=0.001 datamodule=mixing_encoded datamodule.batch_size=512 ~callbacks.visualization_callback logger.wandb.tags=["mila","poly-mixing-disentanglement","fixed-reg"] run_path="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/autoencoder_synthetic_mixing_linear_False_8_6_p2/2023-09-25_18-02-22/"
# p=2, d=8
# python run_training.py ckpt_path=null model=mixing_md_encoded_autoencoder model.optimizer.lr=0.001 datamodule=mixing_encoded datamodule.batch_size=512 ~callbacks.visualization_callback logger.wandb.tags=["mila","poly-mixing-disentanglement","fixed-reg"] run_path="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/autoencoder_synthetic_mixing_linear_False_8_8_p2/2023-09-25_18-58-01/"
# p=2, d=10
# python run_training.py ckpt_path=null model=mixing_md_encoded_autoencoder model.optimizer.lr=0.001 datamodule=mixing_encoded datamodule.batch_size=512 ~callbacks.visualization_callback logger.wandb.tags=["mila","poly-mixing-disentanglement","fixed-reg"] run_path="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/autoencoder_synthetic_mixing_linear_False_8_10_p2/2023-09-25_18-54-54/"
# p=2, d=14
# python run_training.py ckpt_path=null model=mixing_md_encoded_autoencoder model.optimizer.lr=0.001 datamodule=mixing_encoded datamodule.batch_size=512 ~callbacks.visualization_callback logger.wandb.tags=["mila","poly-mixing-disentanglement","fixed-reg"] run_path="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/autoencoder_synthetic_mixing_linear_False_8_14_p2/2023-09-25_18-58-01/"
# p=3, d=6
# python run_training.py ckpt_path=null model=mixing_md_encoded_autoencoder model.optimizer.lr=0.001 datamodule=mixing_encoded datamodule.batch_size=512 ~callbacks.visualization_callback logger.wandb.tags=["mila","poly-mixing-disentanglement","fixed-reg"] run_path="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/autoencoder_synthetic_mixing_linear_False_8_6_p3/2023-09-25_18-42-34/"
# p=3, d=8
# python run_training.py ckpt_path=null model=mixing_md_encoded_autoencoder model.optimizer.lr=0.001 datamodule=mixing_encoded datamodule.batch_size=512 ~callbacks.visualization_callback logger.wandb.tags=["mila","poly-mixing-disentanglement","fixed-reg"] run_path="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/autoencoder_synthetic_mixing_linear_False_8_8_p3/2023-09-25_18-58-01/"
# p=3, d=10
# python run_training.py ckpt_path=null model=mixing_md_encoded_autoencoder model.optimizer.lr=0.001 datamodule=mixing_encoded datamodule.batch_size=512 ~callbacks.visualization_callback logger.wandb.tags=["mila","poly-mixing-disentanglement","fixed-reg"] run_path="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/autoencoder_synthetic_mixing_linear_False_8_10_p3/2023-09-25_18-57-58/"
# p=3, d=14
# python run_training.py ckpt_path=null model=mixing_md_encoded_autoencoder model.optimizer.lr=0.001 datamodule=mixing_encoded datamodule.batch_size=512 ~callbacks.visualization_callback logger.wandb.tags=["mila","poly-mixing-disentanglement","fixed-reg"] run_path="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/autoencoder_synthetic_mixing_linear_False_8_14_p3/2023-09-25_18-58-01/"


# ------------------------------------------------------------------------------------- #
# --------------------------------------- Balls --------------------------------------- #
# ------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------- #
# -------------------------------- Reconstruction Only -------------------------------- #


# cnn with conv and upsampling
# python run_training.py trainer.accelerator='gpu' trainer.devices=1 ckpt_path=null model/optimizer=adamw model.optimizer.lr=0.001 datamodule=md_balls model=balls model.z_dim=64 model/autoencoder=cnn_ae_balls model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["mila","balls","cnn"] ~callbacks.early_stopping

# resnet18
python run_training.py trainer.accelerator='gpu' trainer.devices=1 ckpt_path=null model/optimizer=adamw model.optimizer.lr=0.001 datamodule=md_balls model=balls model.z_dim=25 model/autoencoder=resnet18_ae_balls model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["mila","balls","resnet"] ~callbacks.early_stopping

# python run_training.py trainer.accelerator='cpu' ckpt_path=null model/optimizer=adamw model.optimizer.lr=0.001 datamodule=md_balls model=balls model.z_dim=64 model/autoencoder=resnet18_ae_balls model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["mila","balls","test"] ~callbacks.early_stopping

# ------------------------------------------------------------------------------------- #
# -------------------------- Disentanglement with encoded images ---------------------- #

# -------------------------- min-max penalty, no hinge loss ---------------------- #

# iv=1,sp=1
# cpu
# python run_training.py ckpt_path=null trainer.accelerator='cpu' model.optimizer.lr=0.001 datamodule=balls_encoded model=balls_md_encoded_autoencoder model.z_dim=4 model.z_dim_invariant_fraction=0.5 model.hinge_loss_weight=0.0 model.penalty_criterion="minmax" model.penalty_weight=1.0 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["narval","balls-encoded"] ~callbacks.early_stopping ~callbacks.visualization_callback run_path="/home/aminm/scratch/logs/training/runs/autoencoder_md_balls_64_iv_1_sp_1/2023-09-16_08-23-15" ckpt_path=null
# python run_training.py ckpt_path=null trainer.gradient_clip_val=0.1 trainer.accelerator='cpu' model.optimizer.lr=0.001 datamodule=balls_encoded model=balls_md_encoded_autoencoder model.z_dim=4 model.z_dim_invariant_fraction=0.5 model.hinge_loss_weight=0.0 model.penalty_criterion="minmax" model.penalty_weight=1.0 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["narval","balls-encoded"] ~callbacks.early_stopping ~callbacks.visualization_callback run_path="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/autoencoder_md_balls_64_iv_1_sp_1/2023-09-19_07-33-06" ckpt_path=null

# ------------------------------------------------------------------------------------- #
# --------------------------------------- MNIST --------------------------------------- #
# ------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------- #
# -------------------------- Disentanglement with encoded images ---------------------- #
# python run_training.py trainer.gpus=0 ckpt_path=null model.optimizer.lr=0.001 datamodule=mnist_encoded model=mnist_md_encoded_autoencoder model/autoencoder=mlp_ae_mnist_nc model.z_dim=256 model.z_dim_invariant_fraction=0.9 model.hinge_loss_weight=0.0 model.penalty_criterion="minmax" model.penalty_weight=1.0 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["mila","log"] ~callbacks.early_stopping ~callbacks.visualization_callback run_path="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/autoencoder_multi_domain_mnist_8_256/2023-09-09_07-59-59"
# python run_training.py ckpt_path=null model.optimizer.lr=0.001 datamodule=mnist_encoded model=mnist_md_encoded_autoencoder model.z_dim=256 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["mila","mnist","separate"] ~callbacks.early_stopping ~callbacks.visualization_callback trainer.gpus=1 run_path="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/autoencoder_multi_domain_mnist_8_256/2023-09-09_07-59-59"

# python run_training.py trainer.accelerator='gpu' trainer.devices=1 ckpt_path=null model.optimizer.lr=0.001 datamodule=mnist_encoded model=mnist_md_encoded_autoencoder model/autoencoder=mlp_ae_mnist_nc model.z_dim=256 model.z_dim_invariant_fraction=0.9 model.hinge_loss_weight=0.0 model.penalty_criterion="minmax" model.penalty_weight=1.0 model/scheduler_config=reduce_on_plateau model.scheduler_config.scheduler_dict.monitor="train_loss" logger.wandb.tags=["mila","log"] ~callbacks.early_stopping ~callbacks.visualization_callback run_path="/home/mila/s/sayed.mansouri-tehrani/scratch/logs/training/runs/autoencoder_multi_domain_mnist_8_256/2023-09-09_07-59-59"

conda deactivate
module purge
