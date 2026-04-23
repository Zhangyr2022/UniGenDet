set -euo pipefail
set -x

GPUS=4

output_path=/path/to/project/eval_result/ablation_false_detector_laion_generation

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
    --finetune-path /path/to/project/ablation_bad_detectorresults_train_fakevlm_ema0.99_label_balanced_generation_latent_gen_repa_0_layer/0001000/ema.safetensors
    # --finetune-path /path/to/project/results_train_fakevlm_ema0.99_label_balanced_generation_latent_gen_repa_last_layer/0001400/0001400/ema.safetensors
    # --finetune-path /path/to/project/pretrained/ema.safetensors
    # --finetune-path /path/to/project/results_train_fakevlm_ema0.99_label_balanced_generation_latent/0000800/ema.safetensors
    # --finetune-path /path/to/project/results_train_fakevlm_ema0.99_label_balanced_generation_latent_gen_repa_0_layer/0001500/ema.safetensors

# Stage 2: compute FID/IS from generated results.
python eval/gen/fid_compute.py \
    --generated_base_dir "$output_path/images" \
    --real_image_root /path/to/datasets/Unigendet/laion/