# Fine-tuning
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=6 \
  train/pretrain_unified_navit_gduf.py \
  --dataset_config_file ./data/configs/unigendet_GDUF.yaml \
  --model_path ./pretrained/bagel_7b_mot \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --log_every 1 \
  --lr 2e-5 \
  --num_workers 1 \
  --expected_num_tokens 32768 \
  --max_num_tokens 35000 \
  --max_num_tokens_per_sample 16384 \
  --num_shard 6 \
  --cpu_offload True \
  --ema "0.99" \
  --results_dir "results_gduf" \
  --save_every 200 \
  --checkpoint_dir "gduf" \
  --resume_from ./pretrained/bagel_7b_mot \