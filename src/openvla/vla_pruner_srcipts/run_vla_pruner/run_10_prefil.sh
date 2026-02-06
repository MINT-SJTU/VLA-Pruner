#!/bin/bash
# VLA-Pruner Evaluation Script - LIBERO-10 Task (with Prefill Attention)
# fastv_r: pruning ratio, token retention ratio = 1 - fastv_r

# ============================================
# Experiment 1: Retain 12.5% tokens (fastv_r=0.875)
# ============================================
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --use_text_vision_selection False \
    --use_prefil_attention True \
    --use_temporal True \
    --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
    --use_fastv True \
     --task_suite_name libero_10 \
     --fastv_r 0.875 \
     --run_id_note vlapruner_prefill_12.5% \
     --num_trials_per_task 50

# ============================================
# Experiment 2: Retain 25% tokens (fastv_r=0.75)
# ============================================
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --use_text_vision_selection False \
    --use_prefil_attention True \
    --use_temporal True \
    --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
    --use_fastv True \
    --task_suite_name libero_10 \
    --fastv_r 0.75 \
    --run_id_note vlapruner_prefill_25% \
    --num_trials_per_task 50

# ============================================
# Experiment 3: Retain 50% tokens (fastv_r=0.5)
# ============================================
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --use_text_vision_selection False \
    --use_prefil_attention True \
    --use_temporal True \
    --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
    --use_fastv True \
     --task_suite_name libero_10 \
     --fastv_r 0.5 \
     --run_id_note vlapruner_prefill_50% \
     --num_trials_per_task 50
