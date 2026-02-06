CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --sparsevlm True \
    --use_text_vision_selection True \
    --use_temporal False \
    --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
    --use_fastv True \
    --task_suite_name libero_10 \
    --fastv_r 0.875 \
    --seed 7 \
    --run_id_note sparsevlm_12.5% \
    --num_trials_per_task 50


CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --sparsevlm True \
    --use_text_vision_selection True \
    --use_temporal False \
    --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
    --use_fastv True \
    --task_suite_name libero_10 \
    --fastv_r 0.75 \
    --seed 7 \
    --run_id_note sparsevlm_25% \
    --num_trials_per_task 50



CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --sparsevlm True \
    --use_text_vision_selection True \
    --use_temporal False \
    --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-10 \
    --use_fastv True \
    --task_suite_name libero_10 \
    --fastv_r 0.5 \
    --seed 7 \
    --run_id_note sparsevlm_50% \
    --num_trials_per_task 50
