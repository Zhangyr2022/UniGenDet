<p align="center">
  <h1 align="center">UniGenDet</h1>
  <p align="center"><b>Unified Generative-Discriminative Co-Evolution for Image Generation and Generated Image Detection</b></p>
</p>

<p align="center">
  <a href="https://github.com/Zhangyr2022/UniGenDet"><img src="https://img.shields.io/badge/Code-GitHub-black?logo=github" alt="GitHub"></a>
  <a href="#"><img src="https://img.shields.io/badge/Paper-ArXiv-red?logo=arxiv" alt="Paper"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue" alt="License"></a>
</p>

## Overview

Image generation and generated-image detection have advanced rapidly, but mostly along separate technical trajectories. Generators optimize perceptual realism, while detectors optimize discriminative robustness. This separation creates a persistent lag: detectors overfit to transient artifacts, and generators are not explicitly constrained by forensic criteria.

**UniGenDet** addresses this gap by introducing a unified generative-discriminative framework where generation and detection are co-optimized in a closed loop:

- Generation improves detection by exposing generative logic and reducing distributional blind spots.
- Detection improves generation by feeding authenticity constraints back into synthesis.
- A unified fine-tuning pipeline enables efficient knowledge transfer between both tasks.

This repository is built on top of pretrained BAGEL components and extends them for joint generation-detection co-evolution.

## Highlights

- Unified architecture for **image generation** and **generated-image detection** in one framework.
- Symbiotic multimodal self-attention and detector-informed generative alignment.
- Two practical training modes in this codebase:
  - **DIGA**: detector-informed generative alignment.
  - **GDUF**: generation-detection unified fine-tuning.
- End-to-end evaluation scripts for:
  - Detection benchmarks (`eval/det`)
  - Generation quality and fidelity (`eval/gen`, including FID/IS)

## Method at a Glance

UniGenDet forms a co-evolutionary loop:

1. The generator synthesizes images under multimodal conditions.
2. The detector evaluates authenticity and provides structured supervisory signals.
3. The generator is refined using detector-informed alignment.
4. The detector is updated with richer generative priors and harder synthetic distributions.

This iterative process jointly improves realism, detection generalization, and interpretability.

## Installation

### 1. Clone and create environment

```bash
git clone https://github.com/Zhangyr2022/UniGenDet.git
cd UniGenDet

conda create -n unigendet python=3.10 -y
conda activate unigendet

pip install -r requirements.txt
# Recommended for speed (GPU/CUDA-compatible environment required)
# pip install flash_attn==2.5.8 --no-build-isolation
```

### 2. Prepare pretrained checkpoints

Place required [pretrained BAGEL weights](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT) in `pretrained/` directory.

```bash
# Optional: export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download ByteDance-Seed/BAGEL-7B-MoT --local-dir ./pretrained/bagel_7b_mot --recursive
```

## Data Preparation

### 1. Download datasets

我们分别使用 [Laion](https://huggingface.co/datasets/dclure/laion-aesthetics-12m-umap) 子集和 [FakeClue](https://huggingface.co/datasets/lingcco/FakeClue) 数据集进行训练和评测。请按照以下步骤准备数据：

首先下载laion数据：

```bash
python scripts/data/laion_construction.py
```

而后下载FakeClue数据：

```bash
# Optional: export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download lingcco/FakeClue --local-dir ./datasets/fakeclue --recursive
```

### 2. Configure dataset paths

Edit placeholders in:

- `data/dataset_info.py`
- evaluation scripts under `scripts/eval/*.sh`

Use valid absolute paths on your machine (e.g., `/path/to/datasets/...`).

### 3. Configure dataset mixture

Use task YAMLs in:

- `data/configs/unigendet_DIGA.yaml`
- `data/configs/unigendet_GDUF.yaml`

These files define dataset groups, sampling weights, and image transform settings.

## Training

Before launching, set distributed environment variables and checkpoint paths.

### A. DIGA training (detector-informed generative alignment)

```bash
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  train/pretrain_unified_navit_diga.py \
  --dataset_config_file ./data/configs/unigendet_DIGA.yaml \
  --model_path /path/to/project/pretrained \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --lr 2e-5 \
  --visual_gen True \
  --visual_und False \
  --results_dir results_diga \
  --checkpoint_dir diga
```

### B. GDUF training (generation-detection unified fine-tuning)

```bash
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  train/pretrain_unified_navit_gduf.py \
  --dataset_config_file ./data/configs/unigendet_GDUF.yaml \
  --model_path /path/to/project/pretrained \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --lr 2e-5 \
  --results_dir results_gduf \
  --checkpoint_dir gduf
```

### C. Script-based launching

You can also adapt and run:

```bash
bash scripts/train/train_GDUF.sh
bash scripts/train/train_DIGA.sh
```

### D. Notes

参数定义参考 [BAGEL](https://github.com/ByteDance-Seed/Bagel/blob/main/TRAIN.md#training-config)，仓库默认配置适用于单机8卡(每张卡80GB现存)环境。请根据实际环境调整 `--nnodes`、`--nproc_per_node` 和相关路径参数。若遇到内存不足问题，请适当降低 `--max_num_tokens` 和 `--expected_num_tokens` 等参数。

## Evaluation

### 1. Detection evaluation (FakeVLM)

检测评测脚本位于 `scripts/eval/run_eval_fakevlm.sh`，它将评测UniGenDet在FakeVLM数据集上的检测性能。请确保在运行前正确配置数据路径和模型checkpoint路径。

```bash
bash scripts/eval/run_eval_fakevlm.sh
```

### 2. Generation evaluation on LAION-style prompts

为了评测模型生成真实性，我们使用LAION-style prompts进行生成评测。相关脚本位于 `scripts/eval/run_laion.sh`，请使用与训练集不重合的LAION子集进行评测，以测试模型的泛化能力。

```bash
bash scripts/eval/run_laion.sh
```

### 3. GenEval benchmark

```bash
bash scripts/eval/run_geneval.sh
```

## Key Implementation Notes

- `max_latent_size=64` is recommended for the released BAGEL-based setup.
- Ensure `num_used_data` in YAML is larger than `NUM_GPUS x NUM_WORKERS` for stable sampling.
- If your run is for one branch only:
  - generation-only: set `visual_und=False`
  - detection/understanding-heavy: tune `visual_gen` accordingly
- For debugging and memory-limited hardware, reduce:
  - `expected_num_tokens`
  - `max_num_tokens`
  - `max_num_tokens_per_sample`

## Acknowledgement

This project is built upon the open-source [BAGEL](https://github.com/bytedance-seed/BAGEL) ecosystem and related multimodal tooling. We thank the original authors and community contributors.

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{zhang2026unigendet,
  title   = {UniGenDet: Unified Generative-Discriminative Framework for Co-Evolutionary Image Generation and Generated Image Detection},
  author  = {Author List},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## License

This repository follows the license terms specified in `LICENSE`.
