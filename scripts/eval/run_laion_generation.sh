SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH}"


set -euo pipefail
set -x

GPUS=8

output_path=/path/to/project/eval_result/laion_generation

# Stage 1: generate images.
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    ./eval/gen/gen_laion_mp.py \
    --output_dir $output_path/images \
    --json_path /path/to/datasets/Unigendet/laion/final_metadata.json \
    --batch_size 1 \
    --resolution 1024 \
    --max_latent_size 64 \
    --model-path /path/to/project/pretrained \
    --finetune-path /path/to/project/ema.safetensors

# Stage 2: compute FID/IS from generated results.
python eval/gen/fid_compute.py \
    --generated_base_dir "$output_path/images" \
    --real_image_root /path/to/datasets/Unigendet/laion/