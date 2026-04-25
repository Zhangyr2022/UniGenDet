# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"

set -x

GPUS=8

# TIMEOUT
export NCCL_TIMEOUT=3600 # 3600 1
export TORCH_DIST_TIMEOUT=3600
export NCCL_BLOCKING_WAIT=1

# output_path=/path/to/project/eval_result/geneval_1500_fakevlm_generation_prefreeze_repa_0_layer_model0.99_long
output_path=/path/to/project/eval_result/geneval_base
# generate images
torchrun \
--nnodes=1 \
--node_rank=0 \
--nproc_per_node=$GPUS \
--master_addr=127.0.0.1 \
--master_port=12345 \
./eval/gen/gen_images_mp.py \
--output_dir $output_path/images \
--metadata_file ./eval/gen/geneval/prompts/evaluation_metadata_long.jsonl \
--batch_size 1 \
--num_images 4 \
--resolution 1024 \
--max_latent_size 64 \
--model-path /path/to/project/pretrained \
--finetune-path /path/to/project/results_train_fakevlm_ema0.99_label_balanced_generation_latent_gen_repa_0_layer/0001500/ema.safetensors \
--metadata_file ./eval/gen/geneval/prompts/evaluation_metadata.jsonl \


# calculate score
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    ./eval/gen/geneval/evaluation/evaluate_images_mp.py \
    $output_path/images \
    --outfile $output_path/results.jsonl \
    --model-path ./eval/gen/geneval/model


# summarize score
python ./eval/gen/geneval/evaluation/summary_scores.py $output_path/results.jsonl