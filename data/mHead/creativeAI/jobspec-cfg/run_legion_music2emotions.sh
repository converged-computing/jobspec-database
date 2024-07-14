#!/bin/bash
#SBATCH --job-name=music2emotion
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=6
#SBATCH --partition=cuda
#SBATCH --mem=8GB
#SBATCH --time=06:00:00
#SBATCH --output=mus2emo_%j_out.txt
#SBATCH --error=mus2emo_%j_err.txt

ml purge
#load modules
ml nvidia/cudasdk/10.1
ml intel/python/3/2019.4.088

cd /home/mtesta/ || exit

# %% Dropbox Uploader
#git clone https://github.com/thatbrguy/Dropbox-Uploader.git

#cd ./Dropbox-Uploader/ || exit
#chmod +x dropbox_uploader.sh

#dropbox_uploader.sh
#echo "wZJ15QXvZzMAAAAAAAAAAZabVRJ84LA_mJiIY2-khYD9ZHLY9Ot_1I54QXvaJA_X" > token.txt
#dropbox_uploader.sh
# %%

# if exists remove repo and reclone updated
#rm -Rf creativeAI/
#git clone https://github.com/mHead/creativeAI.git

echo "cloning creativeAI repository DONE!"
echo "moving the dataset into the data_root inside the repository"
mkdir -p /home/mtesta/creativeAI/musicSide_root_data/MusicEmo_dataset_raw_wav

if [ -z "$(ls -A /home/mtesta/creativeAI/musicSide_root_data/MusicEmo_dataset_raw_wav)" ]; then
  echo "The folder: creativeAI/musicSide_root_data/MusicEmo_dataset_raw_wav is empty... going to move the dataset into the data_root inside the repository"
  cp -R /home/mtesta/data/clips_30seconds_preprocessed_BIG /home/mtesta/creativeAI/musicSide_root_data/MusicEmo_dataset_raw_wav
else
  echo "The raw files are already inside the repo path"
fi

cd ./creativeAI || exit
#chmod +x run_legion_music2emotions.sh

cd ./creativeAI || exit

repo_root_legion="/home/mtesta/creativeAI"
code="main.py"

echo "installing modules"
pip install Tornado --user
pip install numpy==1.21.0
pip install tensorflow==2.5.0 --user
pip install ffmpeg==1.4 --user
pip install pydub==0.25.1 --user
pip install scipy --user
pip install torch==1.8.0 --user
pip install torchaudio==0.8.0 --user
pip install torchvision==0.9.0 --user
pip install tensorboard==2.5.0 --user
pip install librosa==0.8.1 --user
pip install seaborn==0.11.1 --user


echo "calling ${code}, with repo_root: ${repo_root_legion}. PWD is: $PWD"


#python3 ${code} -v -r ${repo_root_legion} -raw
python3 ${code} -v -r ${repo_root_legion} -mel
#/Users/head/PycharmProjects/creativeAI/musicSide_root_data/MusicEmo_dataset_raw_wav