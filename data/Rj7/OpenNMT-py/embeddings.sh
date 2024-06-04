#!/bin/bash
#SBATCH --gres=gpu:0       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-02:59
#SBATCH --output=%N-%j.out
#module load miniconda3
source activate pytorch
python tools/embeddings_to_torch.py -emb_file ../embeddings/glovewiki.en.vec -output_file data/embedding.50k.en -dict_file data/tok_en_de_50k.vocab.pt
#python train.py -data data/wmt16_emb -save_model models/wmt16_emb -gpuid 0 -encoder_type 'brnn' -brnn_merge 'concat' -rnn_size 1000 -rnn_type 'GRU' -global_attention 'mlp' -optim 'adadelta' -batch_size 80 -epochs 13 -input_feed
#python translate.py -model models/bahdanau_en_de_acc_57.55_ppl_8.27_e5.pt -src data/newstest2013.tok.en -tgt data/newstest2013.tok.de -output pred.txt -replace_unk -verbose -report_bleu
