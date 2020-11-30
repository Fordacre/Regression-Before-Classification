# !/bin/bash
set -x

mkdir -p logs
mkdir -p results
mkdir -p models
SECONDS=0
#################### script for train/test #######################
stage=train_test_fuse     # train/test/fuse/train_test_fuse
pretrain_dataset=KnetV3  # UCF101/KnetV3

file_name=rbc


method=decouple_ssad
method_temporal=decouple_ssad

mode=temporal  # temporal/spatial
gpu="1"
if [ $stage == train_test_fuse ]
then
    LOG_FILE=logs/${file_name}_${mode}_${pretrain_dataset}_${method}.log
    CUDA_VISIBLE_DEVICES=${gpu} python3 -u ${file_name}.py ${stage} ${pretrain_dataset} ${mode} ${method} ${method_temporal} > ${LOG_FILE} &
else
    CUDA_VISIBLE_DEVICES=${gpu} python3 -u ${file_name}.py ${stage} ${pretrain_dataset} ${mode} ${method} ${method_temporal} &
    
fi

mode=spatial
gpu="0"
if [ $stage == train_test_fuse ]
then
   LOG_FILE=logs/${file_name}_${mode}_${pretrain_dataset}_${method}.log
   CUDA_VISIBLE_DEVICES=${gpu} python3 -u ${file_name}.py ${stage} ${pretrain_dataset} ${mode} ${method} ${method_temporal} > ${LOG_FILE} &
else
   CUDA_VISIBLE_DEVICES=${gpu} python3 -u ${file_name}.py ${stage} ${pretrain_dataset} ${mode} ${method} ${method_temporal} &
fi

wait
# # ####################### script for fuse ##########################
echo "finish training/testing, start fusing"
stage=fuse
python3 -u ${file_name}.py ${stage} ${pretrain_dataset} ${mode} ${method} ${method_temporal}

tRun=$SECONDS
echo "$(($tRun / 60)) minutes and $(($tRun % 60)) seconds elapsed for running."