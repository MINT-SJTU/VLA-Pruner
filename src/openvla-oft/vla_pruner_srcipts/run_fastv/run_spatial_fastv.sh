CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --use_vla_cache False \
    --use_fastv True \
    --use_vla_pruner False \
    --fastv_attention_source prefill \
    --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --fastv_k 3 \
    --fastv_r 0.9275 \
    --seed 7 \
    --run_id_note fastv_7.25% \
    --num_trials_per_task 50


CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --use_vla_cache False \
    --use_fastv True \
    --use_vla_pruner False \
    --fastv_attention_source prefill \
    --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --fastv_k 3 \
    --fastv_r 0.875 \
    --seed 7 \
    --run_id_note fastv_12.5% \
    --num_trials_per_task 50


CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --use_vla_cache False \
    --use_fastv True \
    --use_vla_pruner False \
    --fastv_attention_source prefill \
    --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --fastv_k 3 \
    --fastv_r 0.75 \
    --seed 7 \
    --run_id_note fastv_25% \
    --num_trials_per_task 50
