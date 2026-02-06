#!/bin/bash
# VLA-Pruner Evaluation Script - LIBERO-Goal Task (with FastV selection)
# fastv_r: pruning ratio, token retention ratio = 1 - fastv_r

# ============================================
# Experiment 1: Retain 12.5% tokens (fastv_r=0.875)
# ============================================
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --use_text_vision_selection False \
    --use_temporal True \
    --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
    --use_fastv True \
     --task_suite_name libero_goal \
     --fastv_r 0.875 \
     --seed 7 \
     --run_id_note vlapruner_12.5% \
     --num_trials_per_task 50

# ============================================
# Experiment 2: Retain 25% tokens (fastv_r=0.75)
# ============================================
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --use_text_vision_selection False \
    --use_temporal True \
    --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
    --use_fastv True \
     --task_suite_name libero_goal \
     --fastv_r 0.75 \
     --seed 7 \
     --run_id_note vlapruner_25% \
     --num_trials_per_task 50

# ============================================
# Experiment 3: Retain 50% tokens (fastv_r=0.5)
# ============================================
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --use_text_vision_selection False \
    --use_temporal True \
    --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-goal \
    --use_fastv True \
     --task_suite_name libero_goal \
     --fastv_r 0.5 \
     --seed 7 \
     --run_id_note vlapruner_50% \
     --num_trials_per_task 50
