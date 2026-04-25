#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List

import torch
from PIL import Image
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from safetensors import safe_open
from safetensors.torch import save_file

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from eval.inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UniGenDet unified demo (T2I generation + image-only judgment).")
    parser.add_argument("--mode", type=str, choices=["t2i", "detection"], required=True)

    parser.add_argument("--model_path", type=str, default="./pretrained/bagel_7b_mot")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./model.safetensors",
        help="Supports: single .safetensors, .index.json, checkpoint dir, or comma-separated shards.",
    )

    parser.add_argument("--max_memory_per_gpu", type=str, default="80GiB")
    parser.add_argument("--offload_buffers", action="store_true")

    # t2i args
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--cfg_text_scale", type=float, default=4.0)
    parser.add_argument("--cfg_img_scale", type=float, default=1.5)
    parser.add_argument("--cfg_renorm_min", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="./results/demo")

    # detection args
    parser.add_argument("--image", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.3)

    return parser.parse_args()


def merge_safetensors_files(source_file: str, target_file: str, output_file: str) -> str:
    """Merge missing tensor keys from source checkpoint into target checkpoint."""
    source_tensors = {}
    with safe_open(source_file, framework="pt") as f:
        for key in f.keys():
            source_tensors[key] = f.get_tensor(key)

    target_tensors = {}
    with safe_open(target_file, framework="pt") as f:
        for key in f.keys():
            target_tensors[key] = f.get_tensor(key)

    missing = set(source_tensors.keys()) - set(target_tensors.keys())
    if not missing:
        return target_file

    merged = {**target_tensors, **{k: source_tensors[k] for k in missing}}
    save_file(merged, output_file)
    return output_file


def _write_index_for_shards(shard_files: List[str]) -> str:
    """Build a temporary HF-style index json for raw shard files so accelerate can load them."""
    abs_shards = [os.path.abspath(p) for p in shard_files]
    identity = "|".join(abs_shards)
    digest = hashlib.md5(identity.encode("utf-8")).hexdigest()[:12]

    index_dir = os.path.join("/tmp", f"unigendet_demo_index_{digest}")
    os.makedirs(index_dir, exist_ok=True)

    weight_map: Dict[str, str] = {}
    total_size = 0

    for shard_path in abs_shards:
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Shard file not found: {shard_path}")

        shard_name = os.path.basename(shard_path)
        link_path = os.path.join(index_dir, shard_name)
        if not os.path.exists(link_path):
            os.symlink(shard_path, link_path)

        total_size += os.path.getsize(shard_path)
        with safe_open(shard_path, framework="pt") as f:
            for key in f.keys():
                weight_map[key] = shard_name

    index_path = os.path.join(index_dir, "model.safetensors.index.json")
    with open(index_path, "w", encoding="utf-8") as fp:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, fp)

    return index_path


def resolve_checkpoint_for_dispatch(ckpt_path: str, model_path: str) -> str:
    """Resolve user checkpoint path into a format accepted by load_checkpoint_and_dispatch."""
    raw = ckpt_path.strip()

    # Case A: comma-separated shard list
    if "," in raw:
        shards = [p.strip() for p in raw.split(",") if p.strip()]
        if len(shards) < 2:
            raise ValueError("Comma-separated ckpt_path must contain at least two shard files.")
        return _write_index_for_shards(shards)

    # Case B: direct file path
    if os.path.isfile(raw):
        if raw.endswith(".index.json"):
            return raw
        if raw.endswith(".safetensors"):
            source_ckpt = os.path.join(model_path, "ema.safetensors")
            merged_ckpt = os.path.join(os.path.dirname(raw), "merged_demo.safetensors")
            if os.path.exists(merged_ckpt):
                return merged_ckpt
            if os.path.exists(source_ckpt):
                return merge_safetensors_files(source_ckpt, raw, merged_ckpt)
            return raw
        raise ValueError(f"Unsupported ckpt file: {raw}")

    # Case C: directory path (index or shards inside)
    if os.path.isdir(raw):
        index_json = os.path.join(raw, "model.safetensors.index.json")
        if os.path.exists(index_json):
            return index_json

        shard_files = [
            os.path.join(raw, name)
            for name in sorted(os.listdir(raw))
            if name.endswith(".safetensors")
        ]
        if len(shard_files) == 1:
            return shard_files[0]
        if len(shard_files) >= 2:
            return _write_index_for_shards(shard_files)

        raise FileNotFoundError(f"No safetensors or index found in directory: {raw}")

    raise FileNotFoundError(f"Checkpoint path not found: {raw}")


def build_inferencer(args: argparse.Namespace) -> InterleaveInferencer:
    llm_config = Qwen2Config.from_json_file(os.path.join(args.model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(args.model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=os.path.join(args.model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    n_gpu = max(torch.cuda.device_count(), 1)
    device_map = infer_auto_device_map(
        model,
        max_memory={i: args.max_memory_per_gpu for i in range(n_gpu)},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    same_device_modules = [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
        "vit_pos_embed",
    ]
    first_device = device_map.get(same_device_modules[0], "cuda:0" if torch.cuda.is_available() else "cpu")
    for key in same_device_modules:
        if key in device_map:
            device_map[key] = first_device

    ckpt_for_dispatch = resolve_checkpoint_for_dispatch(args.ckpt_path, args.model_path)

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=ckpt_for_dispatch,
        device_map=device_map,
        offload_buffers=args.offload_buffers,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder="/tmp/offload_unigendet_demo",
    ).eval()

    if isinstance(first_device, str) and first_device.startswith("cuda"):
        vae_model = vae_model.to(first_device).eval()
    else:
        vae_model = vae_model.eval()

    return InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )


def run_t2i(inferencer: InterleaveInferencer, args: argparse.Namespace) -> None:
    if not args.prompt:
        raise ValueError("--prompt is required for --mode t2i")

    os.makedirs(args.output_dir, exist_ok=True)
    output = inferencer(
        text=args.prompt,
        understanding_output=False,
        logits_output=False,
        think=False,
        do_sample=False,
        cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=args.cfg_img_scale,
        num_timesteps=args.num_timesteps,
        cfg_renorm_min=args.cfg_renorm_min,
        image_shapes=(args.resolution, args.resolution),
    )

    if output.get("image") is None:
        raise RuntimeError("T2I failed: inferencer did not return an image.")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"demo_t2i_{ts}.png")
    output["image"].save(out_path)
    print(f"[T2I] Saved image to: {out_path}")


def run_detection(inferencer: InterleaveInferencer, args: argparse.Namespace) -> None:
    if not args.image:
        raise ValueError("--image is required for --mode detection")
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"image not found: {args.image}")

    image = pil_img2rgb(Image.open(args.image))
    prompt = (
        "Does this image look real/fake?"
    )

    output = inferencer(
        image=image,
        text=prompt,
        understanding_output=True,
        logits_output=False,
        use_vae=False,
        think=False,
        do_sample=args.do_sample,
        text_temperature=args.temperature,
        max_think_token_n=args.max_new_tokens,
    )

    print("[detection] Text output:")
    print(output.get("text", ""))

    if output.get("logits") is not None:
        logits = output["logits"][0].detach().float().cpu()
        probs = torch.softmax(logits, dim=-1)
        pred = int(torch.argmax(logits).item())
        print(f"[detection] Logits: {logits.tolist()}")
        print(f"[detection] Probs : {probs.tolist()}")
        print(f"[detection] Pred  : class_{pred}")


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"model_path not found: {args.model_path}")

    inferencer = build_inferencer(args)

    if args.mode == "t2i":
        run_t2i(inferencer, args)
    elif args.mode == "detection":
        run_detection(inferencer, args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
