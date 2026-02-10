# Run VLA-Pruner on OpenVLA

## Relevant Files

**Evaluation Scripts**
* `vla_pruner_srcipts/run_vla_pruner/`: VLA-Pruner evaluation scripts for all LIBERO benchmarks
* `vla_pruner_srcipts/download_model_local.py`: Download checkpoints locally

**Implementation**
* `prismatic/extern/hf/modeling_prismatic.py`: VLA-Pruner inference integration
* `transformers/src/transformers/models/llama/modeling_llama.py`: Token pruning implementation

---

## Setup

Set up a conda environment with LIBERO (follow instructions in [README.md](README.md)).

```bash
conda create -n openvla python=3.10
conda activate openvla
cd src/openvla
pip install -e .
```

---

## VLA-Pruner Evaluation

### Download Checkpoints

```bash
python vla_pruner_srcipts/download_model_local.py \
  --model_id openvla/openvla-7b-finetuned-libero-spatial
```

### Run with VLA-Pruner

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --use_fastv True \
    --use_prefil_attention True \
    --use_temporal True \
    --fastv_r 0.75 \
    --num_trials_per_task 50
```

### Run with FastV

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --use_fastv True \
    --use_temporal False \
    --fastv_r 0.75 \
    --num_trials_per_task 50
```

### Run OpenVLA Baseline

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --use_fastv False \
    --num_trials_per_task 50
```

---

## Evaluation Scripts

All evaluation scripts are located in `vla_pruner_srcipts/run_vla_pruner/`:

| Script | Description |
|--------|-------------|
| `run_spatial.sh` | LIBERO-Spatial with FastV selection + temporal guidance |
| `run_spatial_prefil.sh` | LIBERO-Spatial with prefill selection + temporal guidance |
| `run_object.sh` | LIBERO-Object with FastV selection + temporal guidance |
| `run_object_prefil.sh` | LIBERO-Object with prefill selection + temporal guidance |
| `run_goal.sh` | LIBERO-Goal with FastV selection + temporal guidance |
| `run_goal_prefil.sh` | LIBERO-Goal with prefill selection + temporal guidance |
| `run_10.sh` | LIBERO-10 with FastV selection + temporal guidance |
| `run_10_prefil.sh` | LIBERO-10 with prefill selection + temporal guidance |

**Note:** Scripts with `_prefil` suffix use the prefill attention selection criterion as described in our paper. We also provide scripts without `_prefil` suffix that use FastV's last-token attention selection criterion combined with temporal guidance, which shows better performance in some scenarios.
