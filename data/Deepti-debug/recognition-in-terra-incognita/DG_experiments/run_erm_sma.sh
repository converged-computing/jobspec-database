#!/bin/bash
#SBATCH --job-name=rerm_sma
#SBATCH -A research
#SBATCH -n 38
#SBATCH --gres=gpu:4
#SBATCH -o ../logs/rerm_sma_final2.log
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

cd 
source env_vinet/bin/activate
module load u18/cuda/11.6
module load u18/cudnn/8.4.0-cuda-11.6
a=$USER

cd /home/girmaji08/ensemble-of-averages/rohit_code

rm -rf $ssd_path/rerm_resnet50
mkdir -p $ssd_path/rerm_resnet50
rsync -r $share3_address/terra_incognita $ssd_path
python run_multiple_training.py --data_dir /ssd_scratch/cvit/girmaji08 --output_dir /ssd_scratch/cvit/girmaji08/rerm_resnet50/terra  --n_hparams 2 --n_trials 1  --hparams '{"arch": "resnet50"}' --algorithms 'ERM'


rm -rf $ssd_path/rerm-sma_resnet50
mkdir -p $ssd_path/rerm-sma_resnet50
rsync -r $share3_address/terra_incognita $ssd_path
python run_multiple_training.py --data_dir /ssd_scratch/cvit/girmaji08 --output_dir /ssd_scratch/cvit/girmaji08/rerm-sma_resnet50/terra  --n_hparams 2 --n_trials 1  --hparams '{"arch": "resnet50"}' 

python evaluation.py --data_dir /ssd_scratch/cvit/girmaji08 --dataset TerraIncognita --output_dir /ssd_scratch/cvit/girmaji08/rerm_resnet50/terra --hparams '{"num_workers": 1, "batch_size": 128, "arch": "resnet50"}'

python evaluation.py --data_dir /ssd_scratch/cvit/girmaji08 --dataset TerraIncognita --output_dir /ssd_scratch/cvit/girmaji08/rerm-sma_resnet50/terra --hparams '{"num_workers": 1, "batch_size": 128, "arch": "resnet50"}'

#rsync -r /ssd_scratch/cvit/girmaji08/rerm-sma_resnet50 $share3_address
















# mkdir -p /ssd_scratch/cvit/girmaji08

# mkdir -p $ssd_path/Dataset/annotations
# mkdir -p $ssd_path/Dataset/video_audio
# mkdir -p $ssd_path/Dataset/video_frames


# rsync -r girmaji08@ada.iiit.ac.in:/share3/girmaji08/mvva_dataset/annotations/mvva $ssd_path/Dataset/annotations

# rsync -r girmaji08@ada.iiit.ac.in:/share3/girmaji08/mvva_dataset/video_audio/mvva $ssd_path/Dataset/video_audio

# rsync -r girmaji08@ada.iiit.ac.in:/share3/girmaji08/mvva_dataset/video_frames/mvva $ssd_path/Dataset/video_frames

# rsync -r girmaji08@ada.iiit.ac.in:/share3/girmaji08/mvva_dataset/fold_lists $ssd_path/Dataset/

# rsync -r girmaji08@ada.iiit.ac.in:/share3/girmaji08/mvva_raw_videos $ssd_path/Dataset/

# python train_mvva.py \
# --config configs/AVA/MVVA_eval_SLOWFAST_R50_ACAR_HR2O.yaml \
# --model_val_path /home/girmaji08/ACAR-Net/output/acar_mvva_adam_full_1.pt \
# --videos_root_path $ssd_path/Dataset/mvva_raw_videos \
# --gt_sal_maps_path $ssd_path/Dataset/annotations/mvva \
# --videos_frames_root_path $ssd_path/Dataset/video_frames/mvva \
# --fold_lists_path $ssd_path/Dataset/fold_lists \
#--checkpoint_path /home/girmaji08/ACAR-Net/output/acar_mvva_adam_head_1.pt




#rm -r $ssd_path/test_videos

#mkdir -p $ssd_path/test_videos
#mkdir -p $ssd_path/test_frames
#mkdir -p $ssd_path/gt_sal_root_maps


#rsync -r $share3_address/mvva_raw_videos/294.mp4 $ssd_path/test_videos

#python tools/extract_frames.py --video_dir $ssd_path/test_videos --frame_dir $ssd_path/test_frames 

# get frames

#rsync -r $share3_address/mvva_dataset/video_frames/mvva/294 $ssd_path/test_frames/

# get saliency gt maps
#rsync -r $share3_address/mvva_dataset/annotations/mvva $ssd_path/gt_sal_root_maps 
# get rsync 