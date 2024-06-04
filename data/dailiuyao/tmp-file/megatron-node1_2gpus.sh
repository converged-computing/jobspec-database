#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:09:59
#PBS -q debug
#PBS -l filesystems=home
#PBS -A CSC250STPM09
#PBS -k doe
#PBS -N megatron-test
#PBS -o megatron-test.out
#PBS -e megatron-test.error



source /home/yuke/lyd/conda.sh
conda activate pytorchNCCL

# export $PROTOCOL=RDMA/tcpip/ipoib
export PROTOCOL=RDMA
# export $MODEL=gpt2/bert/t5
export MODEL=bert




export NODE_RANK=0



export MICRO_BATCH_SIZE_GPT2=12

export GLOBAL_BATCH_SIZE_GPT2=24

export MICRO_BATCH_SIZE_BERT=16

export GLOBAL_BATCH_SIZE_BERT=32

export MICRO_BATCH_SIZE_GPT2_L=12

export GLOBAL_BATCH_SIZE_GPT2_L=24

export MICRO_BATCH_SIZE_T5=12

export GLOBAL_BATCH_SIZE_T5=24





cd /home/yuke/lyd/Megatron-LM





rm -rf ./checkpoints/


export GPUS_PER_NODE=4

export NCCL_DEBUG=INFO 

export NNODES=1

export MODEL_PARALLEL_SIZE=2

# export directory="/home/ldai8/bash/Megatron_data_output_profile/${MODEL}/${MODEL_PARALLEL_SIZE}gpus-mbs${MICRO_BATCH_SIZE_BERT}"

# if [ ! -d "$directory" ]; then
#     mkdir -p "$directory/tcpip" "$directory/ipoib" "$directory/RDMA"
#     echo "Directory created."
# else
#     echo "Directory already exists."
# fi


export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


if [ "$MODEL" = "gpt2" ]; then
    export CHECKPOINT_PATH=/home/yuke/lyd/Megatron-LM/checkpoints/gpt2_345m
	export VOCAB_FILE=/home/yuke/lyd/Megatron-LM/model/gpt2-vocab.json
	export MERGE_FILE=/home/yuke/lyd/Megatron-LM/model/gpt2-merges.txt
	export DATA_PATH=/home/yuke/lyd/Megatron-LM/my-gpt2_text_document 
elif [ "$MODEL" = "bert" ]; then
    export CHECKPOINT_PATH=/home/yuke/lyd/Megatron-LM/checkpoints/bert_345m
	export VOCAB_FILE=/home/yuke/lyd/Megatron-LM/model/bert-large-cased-vocab.txt
	export DATA_PATH=/home/yuke/lyd/Megatron-LM/my-bert_text_sentence
elif [ "$MODEL" = "gpt2large" ]; then
   export CHECKPOINT_PATH=/home/yuke/lyd/Megatron-LM/checkpoints/gpt2_774m
	export VOCAB_FILE=/home/yuke/lyd/Megatron-LM/model/gpt2-vocab.json
	export MERGE_FILE=/home/yuke/lyd/Megatron-LM/model/gpt2-merges.txt
	export DATA_PATH=/home/yuke/lyd/Megatron-LM/my-gpt2_text_document  
else
    export CHECKPOINT_PATH=/home/yuke/lyd/Megatron-LM/checkpoints/t5_base
	export VOCAB_FILE=/home/yuke/lyd/Megatron-LM/model/bert-large-cased-vocab.txt
	export DATA_PATH=/home/yuke/lyd/Megatron-LM/my-t5_text_sentence
fi


hostname -I

# # notes for hostname tests
# hsn0 10.201.2.27 ### 100gbps (ib_send_bw 10.201.2.27)
# hsn1 10.201.2.7 ### 100gbps
# bond0 10.140.57.108 ### 100gbps

# hostname -I: all ip address ### 10.140.57.108 10.201.2.7 10.201.2.27

export NAME_ADD=$(hostname -I | awk '{print $NF}')
echo $NAME_ADD

if [ "$PROTOCOL" = "RDMA" ]; then
    export MASTER_ADDR=$NAME_ADD
	export NCCL_SOCKET_IFNAME=hsn0
	export NCCL_NET=IB
elif [ "$PROTOCOL" = "ipoib" ]; then
    export MASTER_ADDR="10.3.1.153"
	export NCCL_SOCKET_IFNAME=ib0
	export NCCL_NET=Socket
else
    export MASTER_ADDR="10.1.1.153"
	export NCCL_SOCKET_IFNAME=en0
	export NCCL_NET=Socket
fi


# export MASTER_ADDR="10.3.1.158"
# export NCCL_SOCKET_IFNAME=ib0
# export NCCL_NET=IB





export MASTER_PORT=21242



export DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
--nnodes $NNODES \
--node_rank $NODE_RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT"

# if [ "$MODEL" = "gpt2" ]; then
#         dool --time --mem --cpu --net -N en0,ib0,lo,total 1 > /home/ldai8/bash/Megatron_data_output_profile/${MODEL}/8gpus-mbs${MICRO_BATCH_SIZE_GPT2}/${PROTOCOL}/${MODEL}-${PROTOCOL}-8gpus-CPU.csv  &
#         nvidia-smi --query-gpu=name,timestamp,uuid,utilization.gpu,memory.total,utilization.memory,power.draw --format=csv -l 1 > /home/ldai8/bash/Megatron_data_output_profile/${MODEL}/8gpus-mbs${MICRO_BATCH_SIZE_GPT2}/${PROTOCOL}/${MODEL}-${PROTOCOL}-8gpus-GPU.csv &
#         sh rtop.sh -d ib0 -p rdma > /home/ldai8/bash/Megatron_data_output_profile/${MODEL}/8gpus-mbs${MICRO_BATCH_SIZE_GPT2}/${PROTOCOL}/${MODEL}-${PROTOCOL}-8gpus-RTOP.csv  &
# elif [ "$MODEL" = "bert" ]; then
#         dool --time --mem --cpu --net -N en0,ib0,lo,total 1 > /home/ldai8/bash/Megatron_data_output_profile/${MODEL}/8gpus-mbs${MICRO_BATCH_SIZE_BERT}/${PROTOCOL}/${MODEL}-${PROTOCOL}-8gpus-CPU.csv  &
#         nvidia-smi --query-gpu=name,timestamp,uuid,utilization.gpu,memory.total,utilization.memory,power.draw --format=csv -l 1 > /home/ldai8/bash/Megatron_data_output_profile/${MODEL}/8gpus-mbs${MICRO_BATCH_SIZE_BERT}/${PROTOCOL}/${MODEL}-${PROTOCOL}-8gpus-GPU.csv &
#         sh rtop.sh -d ib0 -p rdma > /home/ldai8/bash/Megatron_data_output_profile/${MODEL}/8gpus-mbs${MICRO_BATCH_SIZE_BERT}/${PROTOCOL}/${MODEL}-${PROTOCOL}-8gpus-RTOP.csv  &
# elif [ "$MODEL" = "gpt2large" ]; then
#         dool --time --mem --cpu --net -N en0,ib0,lo,total 1 > /home/ldai8/bash/Megatron_data_output_profile/${MODEL}/8gpus-mbs${MICRO_BATCH_SIZE_GPT2_L}/${PROTOCOL}/${MODEL}-${PROTOCOL}-8gpus-CPU.csv  &
#         nvidia-smi --query-gpu=name,timestamp,uuid,utilization.gpu,memory.total,utilization.memory,power.draw --format=csv -l 1 > /home/ldai8/bash/Megatron_data_output_profile/${MODEL}/8gpus-mbs${MICRO_BATCH_SIZE_GPT2_L}/${PROTOCOL}/${MODEL}-${PROTOCOL}-8gpus-GPU.csv &
#         sh rtop.sh -d ib0 -p rdma > /home/ldai8/bash/Megatron_data_output_profile/${MODEL}/8gpus-mbs${MICRO_BATCH_SIZE_GPT2_L}/${PROTOCOL}/${MODEL}-${PROTOCOL}-8gpus-RTOP.csv  &
# else
#         dool --time --mem --cpu --net -N en0,ib0,lo,total 1 > /home/ldai8/bash/Megatron_data_output_profile/${MODEL}/8gpus-mbs${MICRO_BATCH_SIZE_T5}/${PROTOCOL}/${MODEL}-${PROTOCOL}-8gpus-CPU.csv  &
#         nvidia-smi --query-gpu=name,timestamp,uuid,utilization.gpu,memory.total,utilization.memory,power.draw --format=csv -l 1 > /home/ldai8/bash/Megatron_data_output_profile/${MODEL}/8gpus-mbs${MICRO_BATCH_SIZE_T5}/${PROTOCOL}/${MODEL}-${PROTOCOL}-8gpus-GPU.csv &
#         sh rtop.sh -d ib0 -p rdma > /home/ldai8/bash/Megatron_data_output_profile/${MODEL}/8gpus-mbs${MICRO_BATCH_SIZE_T5}/${PROTOCOL}/${MODEL}-${PROTOCOL}-8gpus-RTOP.csv  &
# fi


if [ "$MODEL" = "gpt2" ]; then
    /home/ldai8/data/conda3/envs/pytorchNCCL/bin/python -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_gpt.py \
	--num-layers 24 --hidden-size 1024 --num-attention-heads 16 --seq-length 512 --max-position-embeddings 512 \
	--micro-batch-size $MICRO_BATCH_SIZE_GPT2 --global-batch-size $GLOBAL_BATCH_SIZE_GPT2 --lr 0.00015 --train-iters 1000 --lr-decay-iters 640 \
	--lr-decay-style cosine --vocab-file $VOCAB_FILE --merge-file $MERGE_FILE --lr-warmup-fraction .01 --fp16 \
	--log-interval 10 --save-interval 500 --eval-interval 100 --eval-iters 10 --save $CHECKPOINT_PATH --load $CHECKPOINT_PATH \
	--data-path $DATA_PATH > /home/ldai8/bash/Megatron_data_output_profile/${MODEL}/2gpus-mbs${MICRO_BATCH_SIZE_GPT2}/${PROTOCOL}/${MODEL}-${PROTOCOL}-2gpus-${NODE_RANK}-mbs-${MICRO_BATCH_SIZE_GPT2}.out
elif [ "$MODEL" = "bert" ]; then
    /home/yuke/lyd/conda3/envs/pytorchNCCL/bin/python -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_bert.py \
	--num-layers 24 --hidden-size 1024 --num-attention-heads 16 --seq-length 512 --max-position-embeddings 512 \
	--lr 0.0001 --lr-decay-iters 49 --train-iters 100 --min-lr 0.00001 --lr-warmup-fraction 0.01 \
	--micro-batch-size $MICRO_BATCH_SIZE_BERT --global-batch-size $GLOBAL_BATCH_SIZE_BERT \
	--vocab-file $VOCAB_FILE --split 949,50,1 --fp16 --log-interval 1 --save-interval 50 --eval-interval 10 --eval-iters 1 --recompute-method uniform \
	--save $CHECKPOINT_PATH --load $CHECKPOINT_PATH \
	--data-path $DATA_PATH 
elif [ "$MODEL" = "gpt2large" ]; then
    /home/ldai8/data/conda3/envs/pytorchNCCL/bin/python -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_gpt.py \
	--num-layers 36 --hidden-size 1280 --num-attention-heads 20 --seq-length 512 --max-position-embeddings 512 \
	--micro-batch-size $MICRO_BATCH_SIZE_GPT2_L --global-batch-size $GLOBAL_BATCH_SIZE_GPT2_L --lr 0.00015 --train-iters 1000 --lr-decay-iters 640 \
	--lr-decay-style cosine --vocab-file $VOCAB_FILE --merge-file $MERGE_FILE --lr-warmup-fraction .01 --fp16 \
	--log-interval 10 --save-interval 500 --eval-interval 100 --eval-iters 10 --save $CHECKPOINT_PATH --load $CHECKPOINT_PATH \
	--data-path $DATA_PATH > /home/ldai8/bash/Megatron_data_output_profile/${MODEL}/2gpus-mbs${MICRO_BATCH_SIZE_GPT2_L}/${PROTOCOL}/${MODEL}-${PROTOCOL}-2gpus-${NODE_RANK}-mbs-${MICRO_BATCH_SIZE_GPT2_L}.out
else
    /home/ldai8/data/conda3/envs/pytorchNCCL/bin/python -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_t5.py --num-layers 24 --hidden-size 1024 \
	--num-attention-heads 16 --kv-channels 64 --ffn-hidden-size 3072 --encoder-seq-length 512 --decoder-seq-length 128 --max-position-embeddings 512 \
	--lr 0.0001 --lr-decay-iters 495 --train-iters 1000 --min-lr 0.00001 --lr-warmup-fraction 0.01 \
	--micro-batch-size $MICRO_BATCH_SIZE_T5 --global-batch-size $GLOBAL_BATCH_SIZE_T5 \
	--vocab-file $VOCAB_FILE --vocab-extra-ids 100 --split 949,50,1 --fp16 --log-interval 10 --save-interval 500 --eval-interval 100 --eval-iters 10 \
	--recompute-method uniform --save $CHECKPOINT_PATH --load $CHECKPOINT_PATH \
	--data-path $DATA_PATH > /home/ldai8/bash/Megatron_data_output_profile/${MODEL}/2gpus-mbs${MICRO_BATCH_SIZE_T5}/${PROTOCOL}/${MODEL}-${PROTOCOL}-2gpus-${NODE_RANK}-mbs-${MICRO_BATCH_SIZE_T5}.out
fi








# kill %1
# kill %2
# kill %3



# echo "debug 10"

# dool --time --mem --cpu --net -N en0,en1,ib0,lo,total 1
# sh rtop/rtop.sh -d ib0 -all
# NCCL_IB_DISABLE=1


# python tools/preprocess_data.py \ 
# --input /home/hqi6/data/LLM/Megatron-LM/text/AA/wiki_00 \ 
# --output-prefix my-gpt2 \    
# --vocab ./model/gpt2-vocab.json \       
# --dataset-impl mmap \       
# --tokenizer-type GPT2BPETokenizer \       
# --merge-file ./model/gpt2-merges.txt \  
# --workers 1 \ 
# --chunk-size 1024 \     
# --append-eod 

# python tools/preprocess_data.py --input /home/hqi6/data/LLM/Megatron-LM/text/AA/wiki_00 --output-prefix my-bert --vocab ./model/bert-large-cased-vocab.txt --dataset-impl mmap --tokenizer-type BertWordPieceLowerCase --workers 1 --chunk-size 1024 --split-sentences
