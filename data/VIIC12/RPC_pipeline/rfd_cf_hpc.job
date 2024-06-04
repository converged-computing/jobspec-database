#!/bin/bash
#SBATCH --partition=clara
#SBATCH --job-name=rpc_pipeline
#SBATCH --time=0-03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=10GB
#SBATCH --array=1-1

## [[RFDiffusion->[PMPNN->Consensus->ColabFold]CYCLES ]ROUNDS ]ARRAY - RPC-Pipeline v0.9.0

# _Acknowledgments_
# rmsd.py & sec_structure.py (c) Max Beining
# calculate_consensus.py (c) T. Schiffner

#? Environment settings
PMPNN_PATH=/work/ta905ttoo-rfd/ProteinMPNN
PMPNN_PYTHON=/home/sc.uni-leipzig.de/ta905ttoo/.conda/envs/mlfold/bin/python
PMPNN_ENV=/home/sc.uni-leipzig.de/ta905ttoo/.conda/envs/mlfold/

RDIFF_PATH=/home/sc.uni-leipzig.de/mj334kfyi/software/RFdiffusion
RFDIFF_ENV=/home/sc.uni-leipzig.de/mj334kfyi/.conda/envs/rfdiff

#? RFdiffusion settings
input_pdb="/work/ta905ttoo-rfd/1_rfd/input/input.pdb"
contigs="[20-70/B1-15/20-70/0 A1-243/0]"
inpaint_seq="[B1-3,B5-7,B10,B13,B15]"
length="70-95"

#? Fixed positions in the binder
#* E.g. KSFINKVPSKR -> K=1, S=2 -> Same numbering as in the PDB file, if the chain starts with 1
FIX_POS=(4 8 9 11 12 14)

#? ChainID of the binder (e.g. Antibody) in the input PDB file
BINDER_CHAIN=A

#? ChainID of the target (e.g. Antigen) in the input PDB file
TARGET_CHAIN=B

#? RMSD treshold
#TODO -> if RMSD > rmsd_treshold, abort
#rmsd_treshold=5

#? Sequence of the binder (e.g. Antibody)
target_seq="VQLVESGGGVVQPGGSLRLSCEASGFSFKDYGMHWIRQTPGKGLEWISRISGDTRGTSYVDSVKGRFIVSRDNSRNSLFLQMNSLRSEDTALYYCAALVIVAAGDDFDLWGQGTVVTVSSGSSSSSSSSSSSSSSALTQPLSVSGSPGQSVTISCTGSSSDIGSYNFVSWYRQYPGKAPKVMIYEVNKRPSGVPVRFSGSKSGNTASLTVSGLQHEDEADYYCCSYGGRNNLIFGGGTKLTVL"

#? Which strcucture prediction software should be used? AlphaFold or ColabFold
FOLD=ColabFold
#! AlphaFold does not work with the current setup

#? Project name 
PROJECT_NAME="rfd_pp"

#? How many RFdiffusion designs should be generated?
ROUNDS=2

#? How often should the PMPNN->Consensus->ColabFold cycle be repeated?
CYLCES=2

#? PMPNN settings
chains_to_design="A" # RFD automatically turns the first chain aka binder into A
model="vanilla" # "vanilla" or "hyper"

#? Consensus calculated on the best x percent
#* 10=10%
top_n=10


#! FUNCTIONS
update_info_table() {
    local pdb_name=$1
    local rmsd_value=$2
    local helix_frac=$3
    local sheet_frac=$4
    local loop_frac=$5
    local file="${PROJECT_ARRAY}_results.csv"

    if [ ! -f $file ]; then
        echo "PDB_Name,RMSD,Helix_Fraction,Sheet_Fraction,Loop_Fraction" > $file
    fi

    echo "${pdb_name},${rmsd_value},${helix_frac},${sheet_frac},${loop_frac}" >> $file
}

compute_rmsd_and_secondary_structure() {
    local model=$1
    local rmsd_output=$(python3 ../helper_scripts/rmsd.py --native $input_pdb --native_chain $BINDER_CHAIN --residue_list_native $FIX_POS_RMSD --residue_chain_native $TARGET_CHAIN --model $model --model_chain B --residue_list_model $fixed_positions --residue_chain_model A)
    local rmsd_value=$(echo "$rmsd_output" | grep "RMSD" | awk '{print $2}')

    local ss_output=$(python3 ../helper_scripts/sec_struc.py $model "A")
    local helix_frac=$(echo "$ss_output" | grep "Fraction of Helix" | awk '{print $4}')
    local sheet_frac=$(echo "$ss_output" | grep "Fraction of Sheet" | awk '{print $4}')
    local loop_frac=$(echo "$ss_output" | grep "Fraction of Loops" | awk '{print $4}')

    update_info_table $model $rmsd_value $helix_frac $sheet_frac $loop_frac
}

#!----------SARTING THE PIPELINE----------
version=0.9.9

# Check if FOLD is defined correctly
if [ "$FOLD" != "ColabFold" ] && [ "$FOLD" != "AlphaFold" ]; then
    echo "Error: FOLD must be defined as either 'ColabFold' or 'AlphaFold'."
    exit 1
fi

module purge
module load Anaconda3
source /software/all/Anaconda3/2021.11/etc/profile.d/conda.sh
conda deactivate

# Make x arrays
mkdir -p $PROJECT_NAME
cd $PROJECT_NAME
PROJECT_ARRAY=${PROJECT_NAME}_${SLURM_ARRAY_TASK_ID}
FIX_POS_RMSD=$(IFS=" "; echo "${FIX_POS[*]}")
echo "Thank you for using the RPC-Pipeline v${version}! The project ID is: $PROJECT_ARRAY"

#!-------(ROUND)-------
for rfd_round in $( eval echo {1..$ROUNDS}); do

    echo Current rfd_round Status: $rfd_round of $ROUNDS

    conda activate $RFDIFF_ENV
    module load CUDA/11.1.1-GCC-10.2.0
    module load cuDNN/8.0.4.30-CUDA-11.1.1
    module load GCCcore

    rfd_output=rfd_op/${PROJECT_ARRAY}
    mkdir -p $rfd_output
    RFD_op_name=${PROJECT_ARRAY}_r${rfd_round}

    echo Running RFdiffusion ...
    $RDIFF_PATH/scripts/run_inference.py \
        "inference.output_prefix=$rfd_output/$RFD_op_name/$RFD_op_name" \
        "inference.input_pdb=$input_pdb" \
        "contigmap.contigs=$contigs" \
        "contigmap.inpaint_seq=$inpaint_seq" \
        "contigmap.length=$length" \
        "inference.num_designs=1" \
        "peptide.cyclic=False" >/dev/null
    conda deactivate

    # Calculates the new fixed positions based on the length of the hallucinated binder
    length_hallucination=$(python -c "import numpy as np; data = np.load('${rfd_output}/${RFD_op_name}/${RFD_op_name}_0.trb', allow_pickle=True); print(data['sampled_mask'])" | awk -F'[- ]' '{print substr($1, 3)}')
    echo Length before binder: $length_hallucination
    
    # Add new binder length to every FIX_POS array element
    for i in "${!FIX_POS[@]}"; do
        FIX_POS_NEW[$i]=$(( ${FIX_POS[$i]} + $length_hallucination ))
    done
    # Convert array to string
    fixed_positions=$(IFS=" "; echo "${FIX_POS_NEW[*]}")
    echo New fixed_pos: $fixed_positions

    module purge
    module load Biopython
    module load PyRosetta
    compute_rmsd_and_secondary_structure "${rfd_output}/$RFD_op_name/${RFD_op_name}_0.pdb"
    module purge

    mkdir -p pmpnn_ip/${PROJECT_ARRAY}/

   #!-------(CYLCE)-------
    for pc_cycle in $( eval echo {1..$CYLCES}); do
        echo Starting Cycle $pc_cycle of $CYLCES

        if [ "$pc_cycle" -eq 1 ]; then
            # PDB for first cycle comes from RFD
            cp ${rfd_output}/$RFD_op_name/${RFD_op_name}_0.pdb pmpnn_ip/${PROJECT_ARRAY}/
        fi
 
        #! PMPNN
        module purge
        module load Anaconda3
        source /software/all/Anaconda3/2021.11/etc/profile.d/conda.sh
        conda activate $PMPNN_ENV

		output_pmpnn=pmpnn_op/${PROJECT_ARRAY}/${PROJECT_NAME}_r${rfd_round}_c${pc_cycle}
        mkdir -p $output_pmpnn

		path_for_parsed_chains=$output_pmpnn"/parsed_pdbs.jsonl"
		path_for_assigned_chains=$output_pmpnn"/assigned_pdbs.jsonl"
		path_for_fixed_positions=$output_pmpnn"/fixed_pdbs.jsonl"

		$PMPNN_PYTHON $PMPNN_PATH/helper_scripts/parse_multiple_chains.py --input_path="pmpnn_ip/${PROJECT_ARRAY}" --output_path=$path_for_parsed_chains
        $PMPNN_PYTHON $PMPNN_PATH/helper_scripts/assign_fixed_chains.py --input_path=$path_for_parsed_chains --output_path=$path_for_assigned_chains --chain_list "$chains_to_design"
		$PMPNN_PYTHON $PMPNN_PATH/helper_scripts/make_fixed_positions_dict.py --input_path=$path_for_parsed_chains --output_path=$path_for_fixed_positions --chain_list "$chains_to_design" --position_list "$fixed_positions"
        
        if [ "$model" == "vanilla" ]; then
            custom_model="${PMPNN_PATH}/vanilla_model_weights/v_48_020.pt"
        elif [ "$model" == "hyper" ]; then
            custom_model="../hpmpnn_weights/epoch_last_002-noise.pt"
        fi

        $PMPNN_PYTHON $PMPNN_PATH/protein_mpnn_run_mod.py \
            --jsonl_path $path_for_parsed_chains \
            --chain_id_jsonl $path_for_assigned_chains \
            --out_folder $output_pmpnn \
            --num_seq_per_target 200 \
            --sampling_temp "0.1" \
            --batch_size 1 \
            --omit_AAs "C" \
            --fixed_positions_jsonl $path_for_fixed_positions \
            --custom_model $custom_model >/dev/null
            # --save_score 1 \
            # --save_probs 0
        rm pmpnn_ip/${PROJECT_ARRAY}/*.pdb
        conda deactivate
        module purge
        
        #! Consensus
        mkdir -p "consensus"
        if [ "$FOLD" = "AlphaFold" ]; then
            echo "AF does not work rigth now."
            # python calculate_consensus.py $output_pmpnn/seqs/ consensus/${PROJECT_NAME}_r${rfd_round}_c${pc_cycle}.fa
            # echo ">Target" >> "consensus/${PROJECT_ARRAY}_r${rfd_round}_c${pc_cycle}.fa"
            # echo "$target_seq" >> "consensus/${PROJECT_ARRAY}_r${rfd_round}_c${pc_cycle}.fa"
        else
		    python /work/ta905ttoo-rfd_p/helper_scripts/calculate_consensus.py $output_pmpnn/seqs/ consensus/${PROJECT_ARRAY}_r${rfd_round}_c${pc_cycle}.fa --binder_sequence $target_seq --top $top_n
        fi

#? RFdiffusion settings
        #! Structure prediction
        if [ "$FOLD" = "AlphaFold" ]; then
            #TODO DOES NOT WORK RIGHT NOW
            echo "AlphaFold not avaliable"
            #? Where are the AF weights on the Cluster?
            # Structure prediction with AlphaFold
            # echo "Running AlphaFold"
            
            # module purge
            # module load AlphaFold/2.2.2-foss-2021b-CUDA-11.4.1
            # export ALPHAFOLD_DATA_DIR=/software/databases/alphafold
            # run_alphafold.py --fasta_paths=consensus/${PROJECT_NAME}_r${rfd_round}_c${pc_cycle}.fa \
            #     --max_template_date=2022-01-01 \
            #     --model_preset=multimer \
            #     --output_dir=af_op/ \
            #     --use_gpu_relax=True

            # Copy best model to pmpnn_ip -> for next cycle
            # cp af_op/${PROJECT_NAME}_r${rfd_round}_c${pc_cycle}/${PROJECT_NAME}_r${rfd_round}_c${pc_cycle}/ranked_0.pdb pmpnn_ip/${PROJECT_ARRAY}/${PROJECT_NAME}_r${rfd_round}_c${pc_cycle}.pdb                               
            
            # Copy best model to final_af -> for this round_cylce
            # mkdir -p ${PROJECT_NAME}_best_af_model
            # cp cf_op/${PROJECT_NAME}_r${rfd_round}_c${pc_cycle}/${PROJECT_NAME}_r${rfd_round}_c${pc_cycle}/ranked_0.pdb ${PROJECT_NAME}_best_af_model/${PROJECT_NAME}_r${rfd_round}_c${pc_cycle}.pdb
        elif [ "$FOLD" = "ColabFold" ]; then
            # Structure prediction with ColabFold
            echo "Running ColabFold ..."
            mkdir -p cf_weights/

            module load LocalColabFold
            colabfold_batch --use-gpu-relax --amber --data cf_weights/ consensus/${PROJECT_ARRAY}_r${rfd_round}_c${pc_cycle}.fa cf_op/${PROJECT_ARRAY}/${PROJECT_NAME}_r${rfd_round}_c${pc_cycle}/ &>/dev/null
            
            # Copy best model to pmpnn_ip if not last cycle
            if [ $pc_cycle -lt $CYLCES ]; then
                cp cf_op/${PROJECT_ARRAY}/${PROJECT_NAME}_r${rfd_round}_c${pc_cycle}/A_relaxed_rank_001_alphafold2_multimer_v3_model_*.pdb pmpnn_ip/${PROJECT_ARRAY}/${PROJECT_NAME}_r${rfd_round}_c${pc_cycle}.pdb
            fi
            # Copy best model to final_cf -> for this round_cylce
            mkdir -p ${PROJECT_ARRAY}_best_cf_model
            cp cf_op/${PROJECT_ARRAY}/${PROJECT_NAME}_r${rfd_round}_c${pc_cycle}/A_relaxed_rank_001_alphafold2_multimer_v3_model_*.pdb ${PROJECT_ARRAY}_best_cf_model/${PROJECT_ARRAY}_r${rfd_round}_c${pc_cycle}.pdb
        else
            echo $FOLD not avaliable
        fi

        #! RMSD calculation and secondary structure composition
        if [ "$FOLD" = "AlphaFold" ]; then
            echo "AlphaFold not avaliable"
        elif [ "$FOLD" = "ColabFold" ]; then
            module purge    
            module load PyRosetta
            compute_rmsd_and_secondary_structure "cf_op/${PROJECT_ARRAY}/${PROJECT_NAME}_r${rfd_round}_c${pc_cycle}/A_relaxed_rank_001_alphafold2_multimer_v3_model_*.pdb"
            module purge
        else
            echo $FOLD not avaliable
        fi
    done
    echo "Finished! $pc_cycle of $CYLCES cycles done."
done

# __author__ = "Tom U. Schlegel"
# __contact__ = "tom.schlegel@uni-leipzig.de"
# __license__ = "GNU GPLv3"
# __version__ = "0.9.9"