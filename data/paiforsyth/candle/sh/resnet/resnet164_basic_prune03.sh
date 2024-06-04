#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=20000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-00:30
#SBATCH --output=%N-%jresnet164_basic_prune_03_retrain.out
#module purge
#module load python/3.6.3
#source ~/env1/bin/activate
python -m examples.lab --save_prefix=resnet164_basic_prune_03_retrain --dataset_for_classification=cifar10 --model_type=squeezenet --squeezenet_bypass_first_last  --squeezenet_out_dim=10 --squeezenet_in_channels=3  --squeezenet_mode=next_fire --squeezenet_next_fire_groups=1 --squeezenet_pool_interval_mode=multiply  --squeezenet_base=64 --squeezenet_freq=18 --squeezenet_sr=0.25 --fire_skip_mode=simple --squeezenet_num_fires=54 --squeezenet_pool_interval=18 --squeezenet_conv1_stride=1 --squeezenet_conv1_size=3 --squeezenet_num_conv1_filters=16  --squeezenet_max_pool_size=2 --squeezenet_pooling_count_offset=0   --squeezenet_pool_interval_mode=multiply   --batch_size=128 --num_epochs=200 --optimizer=sgd --sgd_momentum=0.9 --sgd_weight_decay=0.0001  --init_lr=0.2 --lr_scheduler=epoch_anneal --epoch_anneal_numcycles=2   --epoch_anneal_save_last --squeezenet_num_layer_chunks=1 --cifar_random_erase --proxy_context_type=group_prune_context     --save_every_epoch --use_no_grad --report_unpruned  --count_multiplies_every_cycle --fire_skip_mode=zero_pad  --enable_pruning  --prune_target_frac=0.3 --prune_warmup_epochs=100   --prune_unit=12 --cuda # --cuda #--count_multiplies --output_level=debug  #--mod_report #--cuda   



