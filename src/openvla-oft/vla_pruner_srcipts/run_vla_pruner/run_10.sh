#!/bin/bash
# VLA-Pruner Evaluation Script - LIBERO-10 Task
# fastv_r: pruning ratio, token retention ratio = 1 - fastv_r

# ============================================
# Experiment 1: Retain 7.25% tokens (fastv_r=0.9275)
# ============================================
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --use_vla_cache False \
    --use_fastv False \
    --use_vla_pruner True \
    --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-10 \
    --task_suite_name libero_10 \
    --fastv_k 3 \
    --fastv_r 0.9275 \
    --vla_pruner_layer 15 \
    --vla_pruner_mode semantic_action \
    --seed 7 \
    --run_id_note vlapruner_7.25% \
    --num_trials_per_task 50

# ============================================
# Experiment 2: Retain 12.5% tokens (fastv_r=0.875)
# ============================================
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --use_vla_cache False \
    --use_fastv False \
    --use_vla_pruner True \
    --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-10 \
    --task_suite_name libero_10 \
    --fastv_k 3 \
    --fastv_r 0.875 \
    --vla_pruner_layer 15 \
    --vla_pruner_mode semantic_action \
    --seed 7 \
    --run_id_note vlapruner_12.5% \
    --num_trials_per_task 50

# ============================================
# Experiment 3: Retain 25% tokens (fastv_r=0.75)
# ============================================
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --use_vla_cache False \
    --use_fastv False \
    --use_vla_pruner True \
    --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-10 \
    --task_suite_name libero_10 \
    --fastv_k 3 \
    --fastv_r 0.75 \
    --vla_pruner_layer 15 \
    --vla_pruner_mode semantic_action \
    --seed 7 \
    --run_id_note vlapruner_25% \
    --num_trials_per_task 50
