#!/bin/bash

EXP_NAME="l2_setting1"
mkdir -p ./check_points/${EXP_NAME}
echo "1 BCELoss And 0.1 L2Loss" >> ./check_points/${EXP_NAME}/evaluation_log.log

# train stage 1
python train.py \
    --lambdas 1 0.1 \
    --experiment_name ${EXP_NAME} \
    --train_data_root ~/dataset/forgelens/GenImage/train \
    --val_data_root ~/dataset/forgelens/GenImage/val \
    --train_classes stable_diffusion_v_1_4 \
    --val_classes stable_diffusion_v_1_4 \
    --training_stage 1 \
    --stage1_batch_size 16 \
    --stage1_epochs 20 \
    --stage1_learning_rate 0.00005 \
    --stage1_lr_decay_step 2 \
    --stage1_lr_decay_factor 0.7 \
    --Adapter_count 3 \
    --Adapter_reduction_factor 4 \
    --stage2_batch_size 16 \
    --stage2_epochs 10 \
    --stage2_learning_rate 0.0000025 \
    --stage2_lr_decay_step 2 \
    --stage2_lr_decay_factor 0.7 \
    --FAFormer_layers 2 \
    --FAFormer_reduction_factor 1 \
    --FAFormer_head 2 \
    --num_workers 4 \
    --seed 3407

# evaluate stage 1
python evaluate.py \
    --experiment_name ${EXP_NAME} \
    --eval_data_root ~/dataset/forgelens/GenImage/test \
    --eval_stage 1 \
    --Adapter_count 3 \
    --Adapter_reduction_factor 4 \
    --num_workers 4 \
    --seed 3407

EXP_NAME="l2_setting2"
mkdir -p ./check_points/${EXP_NAME}
echo "1 BCELoss And 0.05 L2Loss" >> ./check_points/${EXP_NAME}/evaluation_log.log

# train stage 1
python train.py \
    --lambdas 1 0.05 \
    --experiment_name ${EXP_NAME} \
    --train_data_root ~/dataset/forgelens/GenImage/train \
    --val_data_root ~/dataset/forgelens/GenImage/val \
    --train_classes stable_diffusion_v_1_4 \
    --val_classes stable_diffusion_v_1_4 \
    --training_stage 1 \
    --stage1_batch_size 16 \
    --stage1_epochs 20 \
    --stage1_learning_rate 0.00005 \
    --stage1_lr_decay_step 2 \
    --stage1_lr_decay_factor 0.7 \
    --Adapter_count 3 \
    --Adapter_reduction_factor 4 \
    --stage2_batch_size 16 \
    --stage2_epochs 10 \
    --stage2_learning_rate 0.0000025 \
    --stage2_lr_decay_step 2 \
    --stage2_lr_decay_factor 0.7 \
    --FAFormer_layers 2 \
    --FAFormer_reduction_factor 1 \
    --FAFormer_head 2 \
    --num_workers 4 \
    --seed 3407

# evaluate stage 1
python evaluate.py \
    --experiment_name ${EXP_NAME} \
    --eval_data_root ~/dataset/forgelens/GenImage/test \
    --eval_stage 1 \
    --Adapter_count 3 \
    --Adapter_reduction_factor 4 \
    --num_workers 4 \
    --seed 3407

EXP_NAME="l2_setting3"
mkdir -p ./check_points/${EXP_NAME}
echo "1 BCELoss And 0.01 L2Loss" >> ./check_points/${EXP_NAME}/evaluation_log.log

# train stage 1
python train.py \
    --lambdas 1 0.01 \
    --experiment_name ${EXP_NAME} \
    --train_data_root ~/dataset/forgelens/GenImage/train \
    --val_data_root ~/dataset/forgelens/GenImage/val \
    --train_classes stable_diffusion_v_1_4 \
    --val_classes stable_diffusion_v_1_4 \
    --training_stage 1 \
    --stage1_batch_size 16 \
    --stage1_epochs 20 \
    --stage1_learning_rate 0.00005 \
    --stage1_lr_decay_step 2 \
    --stage1_lr_decay_factor 0.7 \
    --Adapter_count 3 \
    --Adapter_reduction_factor 4 \
    --stage2_batch_size 16 \
    --stage2_epochs 10 \
    --stage2_learning_rate 0.0000025 \
    --stage2_lr_decay_step 2 \
    --stage2_lr_decay_factor 0.7 \
    --FAFormer_layers 2 \
    --FAFormer_reduction_factor 1 \
    --FAFormer_head 2 \
    --num_workers 4 \
    --seed 3407

# evaluate stage 1
python evaluate.py \
    --experiment_name ${EXP_NAME} \
    --eval_data_root ~/dataset/forgelens/GenImage/test \
    --eval_stage 1 \
    --Adapter_count 3 \
    --Adapter_reduction_factor 4 \
    --num_workers 4 \
    --seed 3407