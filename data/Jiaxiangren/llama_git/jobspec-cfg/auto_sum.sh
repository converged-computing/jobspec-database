source ~/miniconda3/etc/profile.d/conda.sh
conda init
conda activate lora


#lora
for dataset in sst-2
do
    CUDA_VISIBLE_DEVICES=0 python Fed_lora.py --lr 7e-6 --path ./flearn/configs/LORA/${dataset}.json | tee ./res/lora/${dataset}.txt
    CUDA_VISIBLE_DEVICES=0 python Fed_lpt.py --lr 7e-6 --path ./flearn/configs/LPT/${dataset}.json | tee ./res/lpt/${dataset}.txt
    CUDA_VISIBLE_DEVICES=0 python Fed_lpt.py --lr 7e-6 --path ./flearn/configs/ATTEMPT/${dataset}.json | tee ./res/attempt/${dataset}.txt
    CUDA_VISIBLE_DEVICES=0 python Fed_lpt.py --lr 7e-6 --path ./flearn/configs/PROMPT/${dataset}.json | tee ./res/prompt/${dataset}.txt
done

for dataset in sst-2
do
    CUDA_VISIBLE_DEVICES=0 python Fed_v2.py --lr 7e-6 --path ./flearn/configs/V2/${dataset}.json | tee ./res/v2/${dataset}.txt 
    CUDA_VISIBLE_DEVICES=0 python Fed_sLora.py --lr 7e-6 --select_method ours --select_layer_num 32 --path ./flearn/configs/ours/${dataset}.json | tee ./res/slora/${dataset}.txt 
    CUDA_VISIBLE_DEVICES=0 python Fed_adaLora.py --lr 7e-6 --select_method ours --select_layer_num 32 --path ./flearn/configs/ours/${dataset}.json | tee ./res/adalora/${dataset}.txt 
    CUDA_VISIBLE_DEVICES=0 python Fed_deltaLora.py --lr 7e-6 --select_method ours --select_layer_num 32 --path ./flearn/configs/ours/${dataset}.json | tee ./res/deltalora/${dataset}.txt
done


for dataset in sst-2
do
    CUDA_VISIBLE_DEVICES=0 python Fed_llama_ours.py --lr 7e-6 --select_method increase --select_layer_num 25 --sort_type voc --path ./flearn/configs/ours/${dataset}.json | tee ./res/voc/${dataset}.txt
    CUDA_VISIBLE_DEVICES=0 python Fed_llama_ours.py --lr 7e-6 --select_method increase --select_layer_num 25 --sort_type seqreo --path ./flearn/configs/ours/${dataset}.json | tee ./res/slw/${dataset}.txt 
    CUDA_VISIBLE_DEVICES=0 python Fed_llama_ours.py --lr 7e-6 --select_method increase --select_layer_num 25 --sort_type shortformer --path ./flearn/configs/ours/${dataset}.json | tee ./res/shortformer/${dataset}.txt
    CUDA_VISIBLE_DEVICES=0 python Fed_llama_ours.py --lr 7e-6 --select_method increase --select_layer_num 32 --sort_type se --path ./flearn/configs/ours/${dataset}.json | tee ./res/se/${dataset}.txt
done




