CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --use_text_vision_selection False \
    --use_temporal False \
    --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
    --use_fastv True \
     --task_suite_name libero_spatial \
     --fastv_r 0.875 \
     --seed 7 \
     --run_id_note fastv_12.5% \
     --num_trials_per_task 50


CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --use_text_vision_selection False \
    --use_temporal False \
    --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
    --use_fastv True \
     --task_suite_name libero_spatial \
     --fastv_r 0.75 \
     --seed 7 \
     --run_id_note fastv_25% \
     --num_trials_per_task 50



CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --use_text_vision_selection False \
    --use_temporal False \
    --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
    --use_fastv True \
     --task_suite_name libero_spatial \
     --fastv_r 0.5 \
     --seed 7 \
     --run_id_note fastv_50% \
     --num_trials_per_task 50
