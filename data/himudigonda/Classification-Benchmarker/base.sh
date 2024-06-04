#! /bin/bash
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -t 3-0
#SBATCH -p general
#SBATCH --gres=gpu:a100:2
#SBATCH -q public
#SBATCH --job-name=SwinV1-B-IN1K-ChestXray14
#SBATCH --output=/scratch/hmudigon/Acad/CSE598-ODL/Supervised/slurm_op/SwinV1-B-IN1K-ChestXray14-%j.out
#SBATCH --error=/scratch/hmudigon/Acad/CSE598-ODL/Supervised/slurm_op/SwinV1-B-IN1K-ChestXray14-%j.err
#SBATCH --mem=80G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hmudigon@asu.edu

# Function to echo the current time
echo_time() {
	echo "Timestamp: [$(/bin/date '+%Y-%m-%d %H:%M:%S')]......................................................$1"
}

echo "===== himudigonda ====="
echo ""
echo ""

echo_time "[1/4] Loading module mamba"
module load mamba/latest
echo_time "[+] Done"
echo ""

echo_time "[2/4] Activating swin virtual environment"
source activate classification-benchmarks
echo_time "[+] Done"
echo ""

echo_time "[3/4] Changing working directory"
cd /scratch/hmudigon/JLiangLab/SOTA/Classification-Benchmarks

echo_time "[+] Done"
echo ""

echo_time "[4/4] Initiating code execution"

python3 main_classification.py \
	--model internimage_base \
	--init imagenet_1k \
	--normalization chestx-ray \
	--data_set MIMIC-CXR \
	--data_dir /scratch/hmudigon/datasets/sl/mimic-cxr-jpg/ \
	--train_list dataset/mimic-cxr-2.0.0-train.csv \
	--val_list dataset/mimic-cxr-2.0.0-validate.csv \
	--test_list dataset/mimic-cxr-2.0.0-test.csv \
	--num_class 14 \
	--batch_size 64 \
	--epochs 200 \
	--exp_name InternImage \
	--lr 0.01 \
	--opt sgd 

	# --data_set MIMIC-CXR \
	# --data_dir /scratch/hmudigon/datasets/sl/mimic-cxr-jpg/ \
	# --train_list dataset/mimic-cxr-2.0.0-train.csv \
	# --val_list dataset/mimic-cxr-2.0.0-validate.csv \
	# --test_list dataset/mimic-cxr-2.0.0-test.csv \
	# --num_class 14 \
	
	# --data_set RSNAPneumonia \
	# --data_dir /scratch/hmudigon/datasets/sl/rsna-pneumonia-detection-challenge/stage_2_train_images_png/ \
	# --train_list dataset/RSNAPneumonia_train.txt \
	# --val_list dataset/RSNAPneumonia_val.txt \
	# --test_list dataset/RSNAPneumonia_test.txt \
	# --num_class 3 \

	# --data_set VinDrCXR \ 
	# --data_dir /scratch/hmudigon/datasets/sl/VinDr-CXR/files/vindr-cxr/1.0.0/ \
	# --train_list dataset/VinDrCXR_train_pe_global_one.txt \
	# --val_list dataset/VinDrCXR_test_pe_global_one.txt \
	# --test_list dataset/VinDrCXR_test_pe_global_one.txt \
	# --num_class 6 \

	# --data_set Shenzhen \
	# --data_dir /scratch/hmudigon/datasets/sl/ShenZhen/images/images/ \
	# --train_list dataset/ShenzenCXR_train_data.txt \
	# --val_list dataset/ShenzenCXR_valid_data.txt \
	# --test_list dataset/ShenzenCXR_test_data.txt \
	# --num_class 1 \

	# 	--data_set CheXpert \
	# --data_dir /scratch/hmudigon/datasets/sl/CheXpert-v1.0/CheXpert-v1.0/images/ \
	# --train_list dataset/CheXpert-v1.0_train.csv \
	# --val_list dataset/CheXpert-v1.0_valid.csv \
	# --test_list dataset/CheXpert-v1.0_valid.csv \
	# --num_class 14 \
	
	# 	--data_set ChestXray14 \
	# --data_dir /scratch/hmudigon/datasets/ssl/ChestXray14/images \
	# --train_list dataset/Xray14_train_official.txt \
	# --val_list dataset/Xray14_val_official.txt \
	# --test_list dataset/Xray14_test_official.txt \
	# --num_class 14 \

echo_time "[+] Done"
echo ""
echo ""

echo_time "[+] Execution completed successfully!"
echo ""
echo "===== himudigonda ====="
