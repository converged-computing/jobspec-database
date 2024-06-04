#!/bin/sh
#SBATCH -o logs/%j.out
#SBATCH -p PV1003q
#SBATCH -n 1
#SBATCH --nodelist=node14


module add cuda11.2/toolkit/11.2.0
source /home/sbmaruf/anaconda3/bin/activate xcodeeval

export CUDA_VISIBLE_DEVICES='1,2'

for i in {3..39..4}; do
python dense_retriever.py \
        model_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-51/dumped_ret_xcodeeval/dpr_biencoder."$i  \
        qa_dataset=XCL_Retrieval_code_code_test \
        ctx_datatsets=[XCL_retrieval_Javascript,XCL_retrieval_Go,XCL_retrieval_Perl,XCL_retrieval_Python,XCL_retrieval_Haskell,XCL_retrieval_Pascal,XCL_retrieval_C++,XCL_retrieval_Scala,XCL_retrieval_D,XCL_retrieval_Rust,XCL_retrieval_C,XCL_retrieval_PHP,XCL_retrieval_Kotlin,XCL_retrieval_Ruby,XCL_retrieval_CS,XCL_retrieval_Java,XCL_retrieval_Ocaml] \
        encoded_ctx_files=[\"/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-51/emb_XCL_retrieval_Javascript_*\",\"/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-51/emb_XCL_retrieval_Go_*\",\"/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-51/emb_XCL_retrieval_Perl_*\",\"/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-51/emb_XCL_retrieval_Python_*\",\"/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-51/emb_XCL_retrieval_Haskell_*\",\"/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-51/emb_XCL_retrieval_Pascal_*\",\"/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-51/emb_XCL_retrieval_C++_*\",\"/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-51/emb_XCL_retrieval_Scala_*\",\"/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-51/emb_XCL_retrieval_D_*\",\"/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-51/emb_XCL_retrieval_Rust_*\",\"/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-51/emb_XCL_retrieval_C_*\",\"/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-51/emb_XCL_retrieval_PHP_*\",\"/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-51/emb_XCL_retrieval_Kotlin_*\",\"/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-51/emb_XCL_retrieval_Ruby_*\",\"/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-51/emb_XCL_retrieval_CS_*\",\"/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-51/emb_XCL_retrieval_Java_*\",\"/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-51/emb_XCL_retrieval_Ocaml_*\"] \
        out_file="/home/sbmaruf/dpr_xcode_eval/outputs/2023-04-28/16-39-51/XCL_Retrieval_code_code_test_ckpt"$i".jsonl"
done

export -n CUDA_VISIBLE_DEVICES

cd /export/home2/sbmaruf/prompt-tuning/prompt-tuning/
sbatch scripts/train/t0_t5-3b/01.mem_prompt_2.sh



