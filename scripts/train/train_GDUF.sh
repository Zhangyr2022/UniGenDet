# Fine-tuning
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/example.yaml \
  --model_path /path/to/project/pretrained \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --log_every 1 \
  --lr 2e-5 \
  --num_worker 1 \
  --expected_num_tokens 32768 \
  --max_num_tokens 35000 \
  --max_num_tokens_per_sample 16384 \
  --num_shard 8 \
  --cpu_offload True \
  --ema "0.99" \
  --results_dir "results_gduf" \
  --save_every 200 \
  --resume-from /path/to/project/pretrained \
  --checkpoint_dir "gduf" \