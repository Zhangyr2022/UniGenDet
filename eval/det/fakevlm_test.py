import os
import json
import argparse
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from PIL import Image
import random
import numpy as np
import traceback

from inferencer import InterleaveInferencer
from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

try:
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError:
    print("Error: missing 'safetensors'. Install with: pip install safetensors")
    exit(1)

try:
    from rouge_score import rouge_scorer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics import f1_score
except ImportError:
    print("Error: missing 'rouge-score' or 'scikit-learn'. Install with: pip install rouge-score scikit-learn")
    exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Run FakeVLM evaluation with multi-GPU workers.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/path/to/project/pretrained",
        help="Path to pretrained model directory.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/path/to/project/results_train_fakevlm_ema0.95_label_balanced_generation_aug/0001200/ema.safetensors",
        help="Path to finetuned checkpoint safetensors file.",
    )
    parser.add_argument(
        "--test_json_path",
        type=str,
        default="/path/to/datasets/fakevlm/data_json/test.json",
        help="Path to FakeVLM test JSON.",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="/path/to/datasets/fakevlm/test",
        help="Root directory for test images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/path/to/project/fakevlm_test_results",
        help="Directory for per-GPU shards and merged final results.",
    )
    parser.add_argument(
        "--final_results_file",
        type=str,
        default="",
        help="Optional final results JSON path. Defaults to <output_dir>/fakevlm_test_final_results.json.",
    )
    parser.add_argument(
        "--logits_output",
        action="store_true",
        help="Use logits argmax as prediction instead of explanation generation.",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Enable sampling during inference.",
    )
    return parser.parse_args()

def merge_safetensors_files(source_file, target_file, output_file):
    """Merge missing tensor keys from source checkpoint into target checkpoint."""
    print(f"Loading source checkpoint: {source_file}")
    source_tensors = {}
    with safe_open(source_file, framework="pt") as f:
        for key in f.keys():
            source_tensors[key] = f.get_tensor(key)

    print(f"Loading target checkpoint: {target_file}")
    target_tensors = {}
    with safe_open(target_file, framework="pt") as f:
        for key in f.keys():
            target_tensors[key] = f.get_tensor(key)

    missing = set(source_tensors.keys()) - set(target_tensors.keys())
    if not missing:
        print("No missing keys found. Target checkpoint is already complete.")
        return target_file

    print(f"Found {len(missing)} missing keys. Merging...")
    merged = {**target_tensors, **{k: source_tensors[k] for k in missing}}
    print(f"Saving merged checkpoint to: {output_file}")
    save_file(merged, output_file)
    print("Merged checkpoint saved successfully.")
    return output_file

def load_model_for_inference(rank: int, args):
    """
        Load model on the single visible GPU (cuda:0) in this subprocess.
        CUDA visibility must be configured before process launch.
    """
    device = "cuda:0"
    print(f"[GPU {rank}] Loading model on {device} ...")

    # Build or reuse merged checkpoint.
    source_ckpt = os.path.join(args.model_path, "ema.safetensors")
    merged_ckpt_path = os.path.join(os.path.dirname(args.ckpt_path), "merged_ema_test.safetensors")
    
    if os.path.exists(merged_ckpt_path):
        print(f"[GPU {rank}] Using existing merged checkpoint: {merged_ckpt_path}")
        final_ckpt_path = merged_ckpt_path
    else:
        print(f"[GPU {rank}] Merging checkpoints...")
        final_ckpt_path = merge_safetensors_files(source_ckpt, args.ckpt_path, merged_ckpt_path)

    # LLM config
    llm_config = Qwen2Config.from_json_file(os.path.join(args.model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    # ViT config
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(args.model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    # Load VAE
    vae_model, vae_config = load_ae(local_path=os.path.join(args.model_path, "ae.safetensors"))

    # Build model config
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    # Tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    device_map = infer_auto_device_map(
        model,
        max_memory={i: "140GiB" for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed'
    ]

    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            device_map[k] = first_device if k in device_map else "cuda:0"
    else:
        first_device = device_map.get(same_device_modules[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    offload_folder = f"/tmp/offload_test_rank_{rank}"
    os.makedirs(offload_folder, exist_ok=True)

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=final_ckpt_path,
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder=offload_folder,
    ).eval()

    vae_model = vae_model.eval()

    print(f"[GPU {rank}] Model loaded successfully.")
    
    return InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )

def test_worker(rank, physical_gpu_id, world_size, data_chunk, progress_queue, args):
    """
    Worker process for one GPU.
    Each process evaluates a data shard and writes a shard-level JSON file.
    """
    # In subprocess scope, the only visible device is logical cuda:0.
    torch.cuda.set_device(0)
    print(f"[GPU {physical_gpu_id}] Worker started (logical rank={rank}).")

    # Per-worker deterministic seed.
    seed = 42 + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        # rank is used to avoid offload folder collisions.
        inferencer = load_model_for_inference(rank, args)
        # Initialize text metrics.
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        vectorizer = TfidfVectorizer(
        )
    except Exception as e:
        print(f"[GPU {physical_gpu_id}] Initialization failed: {e}")
        traceback.print_exc()
        for _ in data_chunk:
            progress_queue.put((rank, None, None))
        return

    # Per-GPU temporary output file.
    temp_output_file = os.path.join(args.output_dir, f"temp_results_gpu_{rank}.json")
    
    results = {}
    correct = 0
    total = 0

    with torch.no_grad():
        for item in data_chunk:
            img_name = "unknown_image"
            img_relative_path = "unknown_path"
            try:
                img_relative_path = item["image"]
                img_name = os.path.basename(img_relative_path)
                img_path = os.path.join(args.image_root, img_relative_path)
                
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Image file not found: {img_path}")

                # Load image and parse labels/response.
                image = Image.open(img_path).convert("RGB")
                
                # Prompt and ground truth.
                prompt = item["conversations"][0]["value"].replace("<image>", "").strip() +"\nExample Output Format:\n        For a real image: \"This is a real image. [Explanation summarizing common realistic features, such as clear outlines, organized objects, realistic shadows, and aligned roads.]\"\n        For a fake image: \"This is a fake image. [Explanation summarizing common unrealistic features, such as blurry outlines, disorganized objects, unrealistic shadows, and misaligned roads.]\"\n"
                # prompt = item["conversations"][0]["value"].replace("<image>", "").strip() +" Be careful to answer whether the image is real or fake, and provide detailed explanations.\n"
                gt_label = item["label"]
                gt_response = item["conversations"][1]["value"]

                # Inference.
                output_dict = inferencer(
                    image=image,
                    text=prompt,
                    understanding_output=True,
                    use_vae=True,
                    logits_output=args.logits_output,
                    think=False,
                    do_sample=args.do_sample,
                )

                # Parse predicted label.
                
                if args.logits_output:
                    logit_predicted_label = predicted_label = output_dict['logits'][0].argmax(-1).item()
                    response_text = f"Logits prediction: {predicted_label}"
                else:
                    response_text = output_dict.get('text', "").lower()
                    predicted_label = -1
                    if "this is a real image" in response_text:
                        predicted_label = 1
                    elif "this is a fake image" in response_text:
                        predicted_label = 0
                    logit_predicted_label = output_dict['logits'][0].argmax(-1).item()
                    
                # Compute ROUGE-L and cosine similarity score (CSS).
                rouge_l_score = 0.0
                css_score = 0.0
                if response_text and gt_response:
                    try:
                        # ROUGE_L
                        rouge_scores = scorer.score(gt_response, response_text)
                        rouge_l_score = rouge_scores['rougeL'].fmeasure

                        # CSS
                        tfidf_matrix = vectorizer.fit_transform([gt_response, response_text])
                        css_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                    except Exception as metric_e:
                        print(f"[GPU {physical_gpu_id}] Metric calculation failed for {img_name}: {metric_e}")


                # Update accuracy counters.
                if predicted_label != -1:
                    total += 1
                    if predicted_label == gt_label:
                        correct += 1

                # Store successful sample result.
                results[img_relative_path] = {
                    "prompt": prompt,
                    "response": output_dict.get('text', response_text) if not args.logits_output else response_text,
                    "predicted_label": predicted_label,
                    "ground_truth_label": gt_label,
                    "ground_truth_response": gt_response,
                    "ROUGE_L": rouge_l_score,
                    "CSS": css_score,
                    "thinking": output_dict.get('thinking', "No thinking process"),
                    "image_path": img_path
                }
                if logit_predicted_label is not None:
                    results[img_relative_path]["logit_predicted_label"] = logit_predicted_label

                print(f"[GPU {physical_gpu_id}] {img_relative_path} - GT: {gt_label}, Pred: {predicted_label}, ROUGE_L: {rouge_l_score:.4f}, CSS: {css_score:.4f}")

            except Exception as e:
                print(f"[GPU {physical_gpu_id}] Failed on {item.get('image', '')}: {e}")
                traceback.print_exc()
                # Store failed sample result.
                results[img_relative_path] = {
                    "prompt": item.get("conversations", [{}])[0].get("value", "prompt").replace("<image>", "").strip(),
                    "error": str(e),
                    "image_path": os.path.join(args.image_root, item.get("image", "")),
                    "predicted_label": -1,
                    "ground_truth_label": item.get("label"),
                    "ROUGE_L": 0.0,
                    "CSS": 0.0,
                }
            finally:
                progress_queue.put((rank, correct, total))

    # Save shard results.
    try:
        with open(temp_output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"[GPU {physical_gpu_id}] Saved shard results to {temp_output_file}")
        print(f"[GPU {physical_gpu_id}] Accuracy: {correct}/{total} = {100*correct/total:.2f}%" if total > 0 else f"[GPU {physical_gpu_id}] No valid predictions")
    except Exception as e:
        print(f"[GPU {physical_gpu_id}] Failed to save shard result file: {e}")
        traceback.print_exc()

def main():
    args = parse_args()

    args.output_dir = os.path.abspath(args.output_dir)
    if not args.final_results_file:
        args.final_results_file = os.path.join(args.output_dir, "fakevlm_test_final_results.json")
    else:
        args.final_results_file = os.path.abspath(args.final_results_file)

    for path_key in ("model_path", "ckpt_path", "test_json_path", "image_root"):
        path_value = getattr(args, path_key)
        if not os.path.exists(path_value):
            print(f"Error: {path_key} does not exist: {path_value}")
            return

    # Global seed.
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if not torch.cuda.is_available():
        print("Error: CUDA is not available.")
        return

    # Resolve target GPU list from CUDA_VISIBLE_DEVICES.
    visible_devices_str = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices_str:
        try:
            target_gpu_ids = [int(x.strip()) for x in visible_devices_str.split(',')]
            print(f"Detected CUDA_VISIBLE_DEVICES list: {target_gpu_ids}")
        except ValueError:
            print(f"Error: invalid CUDA_VISIBLE_DEVICES='{visible_devices_str}'. Expected format like '6,7'.")
            return
    else:
        # Use all visible GPUs.
        target_gpu_ids = list(range(torch.cuda.device_count()))
        print(f"CUDA_VISIBLE_DEVICES is not set. Using all visible GPUs: {target_gpu_ids}")

    world_size = len(target_gpu_ids)
    if world_size == 0:
        print("Error: no GPU selected.")
        return
    
    print(f"Running with {world_size} GPU worker(s).")

    os.makedirs(args.output_dir, exist_ok=True)

    # Resume support: load previously saved final result file.
    existing_results = {}
    processed_items = set()
    if os.path.exists(args.final_results_file):
        print(f"Found existing results file: {args.final_results_file}. Loading for resume.")
        try:
            with open(args.final_results_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            # Mark successfully processed samples.
            for item_key, result_data in existing_results.items():
                if "error" not in result_data:
                    processed_items.add(result_data["image_path"].replace(args.image_root.rstrip("/") + "/", ""))
            print(f"Loaded {len(existing_results)} existing records; {len(processed_items)} successful samples.")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: failed to read existing result file {args.final_results_file}: {e}")
            existing_results = {}
            processed_items = set()
    print("Number of already processed samples:", len(processed_items))

    # Load full test set.
    try:
        with open(args.test_json_path, 'r', encoding='utf-8') as f:
            full_test_data = json.load(f)
        print(f"Loaded JSON with {len(full_test_data)} samples.")
    except Exception as e:
        print(f"Failed to load test JSON: {e}")
        return
    print("full_test_data length:", len(full_test_data))

    # Validate image key uniqueness in metadata.
    full_image_names = [item.get("image", "") for item in full_test_data]
    if len(full_image_names) != len(set(full_image_names)):
        print("Warning: duplicated 'image' keys found in JSON.")
        from collections import Counter
        duplicates = [item for item, count in Counter(full_image_names).items() if count > 1]
        print(f"Duplicate examples: {duplicates[:5]}...")

    if processed_items:
        # Only keep unprocessed samples.
        test_data = [item for item in full_test_data if item.get("image", "") not in processed_items]
        print(f"Skipped {len(full_test_data) - len(test_data)} processed samples. Remaining: {len(test_data)}")
    else:
        test_data = full_test_data

    if not test_data:
        print("No new samples to process.")
        all_results = existing_results
    else:
        # Shuffle for load balancing.
        random.shuffle(test_data)
        # test_data = test_data[:100]  # Optional debug subset.
        print(f"Will process {len(test_data)} samples.")

        # Split evenly across workers.
        chunks = np.array_split(test_data, world_size)
        data_chunks = [c.tolist() for c in chunks if c.size > 0]

        if not data_chunks:
            print("No valid data chunks generated.")
            all_results = existing_results
        else:
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass

            progress_queue = mp.Queue()
            processes = []

            # Save original CUDA_VISIBLE_DEVICES for restoration.
            original_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)

            # Start one worker process per selected physical GPU.
            for rank, physical_gpu_id in enumerate(target_gpu_ids[:len(data_chunks)]):
                os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
                os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

                p = mp.Process(target=test_worker, args=(rank, physical_gpu_id, world_size, data_chunks[rank], progress_queue, args))
                p.start()
                processes.append(p)

            # Restore environment.
            if original_visible is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_visible
            elif "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            # Aggregate progress from workers.
            processed = 0
            with tqdm(total=len(test_data), desc="GPU") as pbar:
                while processed < len(test_data):
                    progress_queue.get()
                    pbar.update(1)
                    processed += 1

            # Wait for worker completion.
            for p in processes:
                p.join()

            print("\nMerging per-GPU result shards...")
            new_results = {}
            # Read and merge all shard files.
            for rank in range(len(data_chunks)):
                temp_file = os.path.join(args.output_dir, f"temp_results_gpu_{rank}.json")
                if os.path.exists(temp_file):
                    try:
                        with open(temp_file, 'r', encoding='utf-8') as f:
                            gpu_results = json.load(f)
                            new_results.update(gpu_results)
                    except Exception as e:
                        print(f"Failed to load shard file {temp_file}: {e}")
            
            # Merge old and new results (new values override on key collision).
            all_results = {**existing_results, **new_results}

    # Save final merged output.
    print(f"Saving {len(all_results)} results to {args.final_results_file}...")
    with open(args.final_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    # Compute summary metrics over valid predictions.
    total_correct = 0
    total_count = 0
    total_rouge_l = 0.0
    total_css = 0.0
    metric_count = 0
    all_gts = []
    all_preds = []
    error_count = 0

    for img_name, v in all_results.items():
        if "error" in v:
            error_count += 1
            continue

        # Accuracy counters.
        if v.get('predicted_label', -1) != -1:
            total_count += 1
            if v['predicted_label'] == v.get('ground_truth_label'):
                total_correct += 1
            
            # F1 inputs.
            all_gts.append(v.get('ground_truth_label'))
            all_preds.append(v.get('predicted_label'))
        
        # ROUGE/CSS accumulators.
        if 'ROUGE_L' in v and 'CSS' in v:
            total_rouge_l += v['ROUGE_L']
            total_css += v['CSS']
            metric_count += 1

    avg_rouge_l = (total_rouge_l / metric_count) if metric_count > 0 else 0.0
    avg_css = (total_css / metric_count) if metric_count > 0 else 0.0
    
    # Macro-F1 score.
    f1 = f1_score(all_gts, all_preds, average='macro') if len(all_gts) > 0 else 0.0

    print(f"\n=== Final Summary ===")
    print(f"Total samples in JSON: {len(full_test_data)}")
    print(f"Saved result records: {len(all_results)}")
    print(f"Samples with errors: {error_count}")
    print(f"Valid predictions: {total_count}")
    print(f"Correct predictions: {total_correct}")
    print(f"Accuracy: {100*total_correct/total_count:.2f}%" if total_count > 0 else "Accuracy: N/A")
    print(f"Macro F1: {f1:.4f}")
    print(f"Average ROUGE_L: {avg_rouge_l:.4f}")
    print(f"Average CSS: {avg_css:.4f}")
    print(f"\nResults saved to: {args.final_results_file}")

if __name__ == "__main__":
    main()