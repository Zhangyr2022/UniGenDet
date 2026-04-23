import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, PngImagePlugin
from pytorch_fid import fid_score
from scipy.stats import entropy
from torchvision.models import inception_v3

# Allow larger PNG text/ICC chunks to avoid decompression errors on metadata-heavy files.
PngImagePlugin.MAX_TEXT_CHUNK = 10 * 1024 * 1024 # 10 MB


def resize_and_save(image_path: str, output_path: str, size=(299, 299)) -> None:
    """Center-crop to a square and resize for Inception-based metrics."""
    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")

            width, height = image.size
            if width != height:
                short_side = min(width, height)
                left = (width - short_side) / 2
                top = (height - short_side) / 2
                right = (width + short_side) / 2
                bottom = (height + short_side) / 2
                image = image.crop((left, top, right, bottom))

            if image.size != size:
                image = image.resize(size, Image.LANCZOS)

            image.save(output_path)
    except Exception as exc:
        print(f"Warning: failed to resize image {image_path}: {exc}")


def calculate_inception_score(image_dir: str, device: torch.device, batch_size: int = 50, splits: int = 10):
    """Compute Inception Score (IS) for generated images."""
    print("\nCalculating Inception Score (IS)...")

    image_paths = [
        os.path.join(image_dir, name)
        for name in os.listdir(image_dir)
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    num_images = len(image_paths)
    if num_images == 0:
        print("Warning: no images found for IS calculation.")
        return 0.0, 0.0

    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()

    preds = []
    for start in range(0, num_images, batch_size):
        batch_paths = image_paths[start:start + batch_size]
        batch_images = []
        for path in batch_paths:
            image = Image.open(path).convert("RGB")
            np_img = np.array(image)
            tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0
            batch_images.append(tensor_img)

        batch = torch.stack(batch_images).to(device)
        with torch.no_grad():
            pred = inception_model(batch)
            pred = F.softmax(pred, dim=1).cpu().numpy()
            preds.append(pred)

    preds = np.concatenate(preds, axis=0)

    split_size = max(1, num_images // splits)
    scores = []
    for split_idx in range(splits):
        part = preds[split_idx * split_size:(split_idx + 1) * split_size, :]
        if part.shape[0] == 0:
            continue
        py = np.mean(part, axis=0)
        kl_divs = [entropy(pyx, py) for pyx in part]
        scores.append(np.exp(np.mean(kl_divs)))

    is_mean = float(np.mean(scores)) if scores else 0.0
    is_std = float(np.std(scores)) if scores else 0.0
    print(f"Inception Score (IS): {is_mean:.2f} +/- {is_std:.2f}")
    return is_mean, is_std


def collect_pairs_and_prepare_images(generated_base_dir: str, real_image_root: str, tmp_real_dir: str, tmp_gen_dir: str) -> None:
    """Collect generated/real image pairs and normalize them into temporary folders."""
    subfolders = [entry.path for entry in os.scandir(generated_base_dir) if entry.is_dir()]
    print(f"Found {len(subfolders)} generation folders.")

    for folder in subfolders:
        try:
            gen_images = glob.glob(os.path.join(folder, "*.png"))
            if not gen_images:
                print(f"Warning: no PNG image found in {folder}, skipping.")
                continue
            generated_image_path = gen_images[0]

            metadata_path = os.path.join(folder, "metadata.json")
            with open(metadata_path, "r", encoding="utf-8") as fp:
                metadata = json.load(fp)

            relative_real_path = metadata.get("image_path")
            if not relative_real_path:
                print(f"Warning: missing 'image_path' in {metadata_path}, skipping.")
                continue

            real_image_path = os.path.join(real_image_root, relative_real_path)
            if not os.path.exists(real_image_path):
                print(f"Warning: real image does not exist: {real_image_path}, skipping.")
                continue

            folder_name = os.path.basename(folder)
            new_real_image_path = os.path.join(tmp_real_dir, f"{folder_name}_real.png")
            new_generated_image_name = f"{folder_name}_{os.path.basename(generated_image_path)}"
            new_generated_image_path = os.path.join(tmp_gen_dir, new_generated_image_name)

            resize_and_save(real_image_path, new_real_image_path)
            resize_and_save(generated_image_path, new_generated_image_path)
        except Exception as exc:
            print(f"Warning: failed to process folder {folder}: {exc}")


def calculate_fid_and_is(generated_base_dir: str, real_image_root: str, batch_size: int = 50) -> None:
    """Calculate FID between generated and real images, then calculate IS on generated images."""
    tmp_real_dir = "temp_real_images_for_fid"
    tmp_gen_dir = "temp_generated_images_for_fid"

    for path in (tmp_real_dir, tmp_gen_dir):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    print(f"Temporary folders created: {tmp_real_dir}, {tmp_gen_dir}")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    collect_pairs_and_prepare_images(generated_base_dir, real_image_root, tmp_real_dir, tmp_gen_dir)

    num_real = len(os.listdir(tmp_real_dir))
    num_gen = len(os.listdir(tmp_gen_dir))
    print(f"Prepared real images: {num_real}")
    print(f"Prepared generated images: {num_gen}")

    print("\nCalculating FID...")
    try:
        fid_value = fid_score.calculate_fid_given_paths(
            paths=[tmp_real_dir, tmp_gen_dir],
            batch_size=batch_size,
            device=device,
            dims=2048,
        )
        print(f"FID: {fid_value:.2f}")
    except Exception as exc:
        print(f"Error during FID calculation: {exc}")

    try:
        calculate_inception_score(image_dir=tmp_gen_dir, device=device, batch_size=batch_size)
    except Exception as exc:
        print(f"Error during IS calculation: {exc}")
    finally:
        print("\nCleaning temporary folders...")
        shutil.rmtree(tmp_real_dir)
        shutil.rmtree(tmp_gen_dir)
        print("Done.")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute FID/IS for generated LAION samples.")
    parser.add_argument(
        "--generated_base_dir",
        type=str,
        default="/path/to/project/eval_result/ablation_false_detector_laion_generation/images",
        help="Folder containing generation subfolders with image and metadata.json.",
    )
    parser.add_argument(
        "--real_image_root",
        type=str,
        default="/path/to/datasets/Unigendet/laion/",
        help="Root folder for real images referenced by metadata['image_path'].",
    )
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size used in FID/IS passes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    generated_base_dir = str(Path(args.generated_base_dir))
    real_image_root = str(Path(args.real_image_root))

    calculate_fid_and_is(
        generated_base_dir=generated_base_dir,
        real_image_root=real_image_root,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
