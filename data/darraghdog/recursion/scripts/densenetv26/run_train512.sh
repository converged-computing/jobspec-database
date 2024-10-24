N_GPU=1
WDIR='densenetv26'
FOLD=0

bsub  -q lowpriority -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=docker.optum.com/dhanley2/bert:cgan_build \
            -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/recursion/scripts/$WDIR  && python3 train.py  \
            --logmsg recursion-pytorch-onecycle  --nbags 20  --epochs 70 --fold $FOLD  --lr 0.0001 --batchsize 16 --workpath scripts/$WDIR  \
            --cutmix_prob 1.0 --beta 1.0  --imgpath data/mount/512X512X6/ --weightsname weights/pytorch_model_densenet$FOLD.bin"


#bsub  -q lowpriority -gpu "num=$N_GPU:mode=exclusive_process" -app gpu -n =$N_GPU  -env LSB_CONTAINER_IMAGE=docker.optum.com/dhanley2/bert:cgan_build \
#            -n 1 -R "span[ptile=4]" -o log_train_%J  sh -c "cd /share/dhanley2/recursion/scripts/$WDIR  && python3 train.py  \
#            --nbags 5  --epochs 150 --fold $FOLD  --lr 0.0001 --batchsize 16 --workpath scripts/$WDIR  \
#            --cutmix_prob 1.0 --beta 1.0  --imgpath data/mount/256X256X6/ --weightsname weights/pytorch_model_densenet$FOLD.bin"
