# diffu_grpo

GRPO training for dUltra with **DeepSpeed ZeRO-3** (`accelerate.yaml`).

## Quick start

1. Install deps from the repo root (PyTorch, `transformers`, `trl`, `accelerate`, `peft`, `deepspeed`, `huggingface_hub`).
2. Point `MODEL_PATH` at a local checkpoint directory or a Hugging Face repo id.
3. Launch on a 4- or 8-GPU node:

```bash
export MODEL_PATH="/path/to/checkpoint-or-hf-repo"   # local dir or HF repo id
bash run_grpo_gsm.sh
```

Slurm: `sbatch run_grpo_gsm.sbatch` (`#SBATCH --gres=gpu:8`; the inner script also supports 4 GPUs).

Outputs land under `./checkpoints/<RUN_NAME>/`; training logs go to `./logs/`.

## Layout

- `run_grpo_gsm.sh` / `.sbatch` — entry point (call chain: script → `diffu_grpo_train.py` → `diffu_grpo_trainer.py`).
- `diffu_grpo_config.py` — all `--flag` defaults (`DiffuGRPOConfig` extends `trl.GRPOConfig`).
- `accel_reward.py`, `reward_func.py` — reward functions.
- `data_utils.py` — dataset loaders.
- `accelerate.yaml`, `sbatch_scripts/` — launch configs and alternate launchers.

## Knobs

- `MODEL_PATH` (env, required).
- Other hyperparameters: edit the top of `run_grpo_gsm.sh`.

## Notes

- `gen_step_reward` weight is **0.0** when `--freeze_unmasking_head true`, else **0.1**.
- `verifier_distillation_reward` weight is **1.0**; the teacher model is chosen per-sample (math vs coding). Optionally set `--teacher_device cuda:0` to pin it.
- If `MODEL_PATH` is an HF repo id, rank 0 calls `snapshot_download` first; the other ranks then hit the cache after the barrier.

PR: [dUltra-os#2](https://github.com/chinsengi/dUltra-os/pull/2)
