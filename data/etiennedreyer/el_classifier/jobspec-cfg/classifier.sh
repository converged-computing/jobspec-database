# CROSS-VALIDATION TRAINING (array jobs)
n_e=15e6; n_epochs=100; n_classes=2; n_tracks=5; n_folds=10; verbose=2; scalars=ON; images=ON; NN_type=CNN
fold=$VAR; output_dir=outputs
python classifier.py  --n_train=$n_e          --n_valid=0           --batch_size=5e3     --n_epochs=$n_epochs  \
                      --n_classes=$n_classes  --n_tracks=$n_tracks  --verbose=$verbose   --dropout=0.05        \
                      --images=$images        --scalars=$scalars    --NN_type=$NN_type   --weight_type='none'  \
                      --train_cuts '(sample["eventNumber"]%'${n_folds}'!='$(($fold-1))')'                      \
                      --valid_cuts '(sample["eventNumber"]%'${n_folds}'=='$(($fold-1))')'                      \
                      --output_dir=$output_dir  --scaler_out scaler_${fold}.pkl --model_out model_${fold}.h5


exit


################################################################################################################
#### PICK A SHELL SCRIPT FROM BELOW ############################################################################
################################################################################################################


# SINGLE TRAINING
python classifier.py  --n_train=10e6  --n_valid=1e6  --batch_size=5e3  --n_epochs=100  --n_classes=2           \
                      --n_tracks=5    --l2=1e-8      --dropout=0.05    --verbose=2     --NN_type=CNN           \
		      --plotting=ON   --weight_type=None --output_dir=outputs


# TRAINING (array jobs)
python classifier.py  --n_train=10e6      --n_valid=15e6      --batch_size=5e3   --n_epochs=100  --n_classes=2 \
                      --n_tracks=${VAR}   --l2=1e-8           --dropout=0.05     --verbose=2     --NN_type=CNN \
                      --plotting=OFF      --weight_type=None  --scalars=ON       --FCN_neurons 200 200         \
                      --output_dir  outputs                   --model_out   model_${VAR}-tracks.h5             \
                      --scaler_out  scaler_${VAR}-tracks.pkl  --results_out results_${VAR}-tracks.pkl


# CROSS-VALIDATION TRAINING (array jobs)
n_e=15e6; n_epochs=100; n_classes=2; n_tracks=$VAR; n_folds=10; verbose=2; scalars=ON; images=ON; NN_type=CNN
fold=$VAR; output_dir=outputs/${VAR}-track    #${SCRIPT_VAR}_to_1
for ((fold = 1; fold <= $n_folds; fold++)) do #for fold in 1 2 3 4 5 do
python classifier.py  --n_train=$n_e          --n_valid=0           --batch_size=5e3     --n_epochs=$n_epochs  \
                      --n_classes=$n_classes  --n_tracks=$n_tracks  --verbose=$verbose   --dropout=0.05        \
                      --images=$images        --scalars=$scalars    --NN_type=$NN_type   --weight_type='none'  \
                      --train_cuts '(sample["eventNumber"]%'${n_folds}'!='$(($fold-1))')'                      \
                      --valid_cuts '(sample["eventNumber"]%'${n_folds}'=='$(($fold-1))')'                      \
                      --output_dir=$output_dir  --scaler_out scaler_${fold}.pkl --model_out model_${fold}.h5   \
                      --results_out results_${fold}.pkl  #--sbatch_var=$SCRIPT_VAR
done
python classifier.py  --n_train=0           --n_valid=$n_e        --n_epochs=0        --n_classes=$n_classes   \
                      --n_tracks=$n_tracks  --n_folds=${n_folds}  --NN_type=$NN_type  --images=$images         \
                      --scalars=$scalars    --output_dir=$output_dir   --results_out=valid_probs.pkl


# USING VALIDATION RESULTS FOR PLOTTING
python classifier.py  --n_valid=15e6  --output_dir=outputs  --results_in=valid_probs.pkl  --plotting=ON
