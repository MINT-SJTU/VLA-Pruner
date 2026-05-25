# Run VLA-Pruner on OpenVLA-OFT

## Relevant Files

**Evaluation Scripts**
* `vla_pruner_srcipts/run_vla_pruner/`: VLA-Pruner evaluation scripts for all LIBERO benchmarks
* `vla_pruner_srcipts/run_fastv/`: FastV baseline evaluation scripts for all LIBERO benchmarks
* `vla_pruner_srcipts/download_model_local.py`: Download checkpoints locally

**Implementation**
* `prismatic/extern/hf/modeling_prismatic.py`: OpenVLA-OFT inference integration
* `transformers/src/transformers/models/llama/modeling_llama.py`: Token pruning implementation

---

## Setup

Set up a conda environment with LIBERO (follow instructions in [README.md](README.md)).

```bash
conda create -n openvla-oft python=3.10
conda activate openvla-oft
cd src/openvla-oft
pip install -e transformers
pip install -e .
```

---

## VLA-Pruner Evaluation

### Download Checkpoints

```bash
python vla_pruner_srcipts/download_model_local.py \
  --model_id moojink/openvla-7b-oft-finetuned-libero-spatial
```

### Run with VLA-Pruner

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --use_vla_cache False \
    --use_fastv False \
    --use_vla_pruner True \
    --fastv_k 3 \
    --fastv_r 0.75 \
    --vla_pruner_layer 15 \
    --vla_pruner_mode semantic_action \
    --num_trials_per_task 50
```

### Run with FastV

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --use_vla_cache False \
    --use_fastv True \
    --use_vla_pruner False \
    --fastv_k 3 \
    --fastv_r 0.75 \
    --fastv_attention_source prefill \
    --num_trials_per_task 50
```

### Run OpenVLA-OFT Baseline

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint checkpoints/openvla-7b-oft-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --use_vla_cache False \
    --use_fastv False \
    --use_vla_pruner False \
    --num_trials_per_task 50
```

---

## Evaluation Scripts

All evaluation scripts are located in `vla_pruner_srcipts/`:

| Script | Description |
|--------|-------------|
| `run_vla_pruner/run_spatial.sh` | LIBERO-Spatial with prefill selection + historical action guidance |
| `run_vla_pruner/run_object.sh` | LIBERO-Object with prefill selection + historical action guidance |
| `run_vla_pruner/run_goal.sh` | LIBERO-Goal with prefill selection + historical action guidance |
| `run_vla_pruner/run_10.sh` | LIBERO-10 with prefill selection + historical action guidance |
| `run_fastv/run_spatial_fastv.sh` | LIBERO-Spatial with FastV prefill selection |
| `run_fastv/run_object_fastv.sh` | LIBERO-Object with FastV prefill selection |
| `run_fastv/run_goal_fastv.sh` | LIBERO-Goal with FastV prefill selection |
| `run_fastv/run_10_fastv.sh` | LIBERO-10 with FastV prefill selection |

**Note:** `fastv_r` is the pruning ratio, so token retention ratio is `1 - fastv_r`. The scripts evaluate 7.25%, 12.5%, and 25% retained visual tokens with `fastv_r=0.9275`, `0.875`, and `0.75`.
