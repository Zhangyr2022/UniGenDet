import os
import json
import shutil
import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, PngImagePlugin
from pytorch_fid import fid_score
from scipy.stats import entropy
from torchvision.models import inception_v3

# Increase PNG text chunk limit to avoid ICC/text metadata parse errors.
PngImagePlugin.MAX_TEXT_CHUNK = 10 * 1024 * 1024  # 10 MB


def resize_and_save(image_path, output_path, size=(299, 299)):
    """Center-crop to square, resize to target resolution, and save as RGB."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")

            width, height = img.size
            if width != height:
                short_side = min(width, height)
                left = (width - short_side) / 2
                top = (height - short_side) / 2
                right = (width + short_side) / 2
                bottom = (height + short_side) / 2
                img = img.crop((left, top, right, bottom))

            if img.size != size:
                img = img.resize(size, Image.LANCZOS)

            img.save(output_path)
    except Exception as e:
        print(f"Failed to process image {image_path}: {e}")


def calculate_inception_score(image_dir, device, batch_size=50, splits=10):
    """Compute Inception Score (IS) for generated images."""
    print("\nComputing Inception Score (IS)...")

    image_paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    num_images = len(image_paths)
    if num_images == 0:
        print("No images found for IS computation.")
        return 0.0, 0.0

    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()

    preds = []
    for i in range(0, num_images, batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_images = []
        for path in batch_paths:
            img = Image.open(path).convert("RGB")
            np_img = np.array(img)
            tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0
            batch_images.append(tensor_img)

        batch = torch.stack(batch_images).to(device)

        with torch.no_grad():
            pred = inception_model(batch)
            pred = F.softmax(pred, dim=1).cpu().numpy()
            preds.append(pred)

    preds = np.concatenate(preds, 0)

    scores = []
    split_size = max(1, num_images // splits)
    for i in range(splits):
        start = i * split_size
        end = min((i + 1) * split_size, num_images)
        if start >= end:
            continue

        part = preds[start:end, :]
        py = np.mean(part, axis=0)
        kl_divs = []
        for k in range(part.shape[0]):
            pyx = part[k, :]
            kl_divs.append(entropy(pyx, py))
        scores.append(np.exp(np.mean(kl_divs)))

    if not scores:
        return 0.0, 0.0

    is_mean = np.mean(scores)
    is_std = np.std(scores)

    print(f"Inception Score (IS): {is_mean:.2f} ± {is_std:.2f}")
    return is_mean, is_std


def calculate_fid_and_is():
    """Build aligned real/generated sets, then compute FID and IS."""
    generated_base_dir = (
        "/path/to/project/eval_result/ablation_false_detector_laion_generation/images"
    )
    real_image_root = "/path/to/datasets/Unigendet/laion/"

    tmp_real_dir = "temp_real_images_for_fid"
    tmp_gen_dir = "temp_generated_images_for_fid"

    if os.path.exists(tmp_real_dir):
        shutil.rmtree(tmp_real_dir)
    if os.path.exists(tmp_gen_dir):
        shutil.rmtree(tmp_gen_dir)

    os.makedirs(tmp_real_dir)
    os.makedirs(tmp_gen_dir)

    print(f"Temporary folders prepared: '{tmp_real_dir}', '{tmp_gen_dir}'")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU not available. Falling back to CPU.")

    subfolders = [f.path for f in os.scandir(generated_base_dir) if f.is_dir()]

    print(f"Found {len(subfolders)} generation subfolders.")
    for folder in subfolders:
        try:
            gen_images = glob.glob(os.path.join(folder, "*.png"))
            if not gen_images:
                print(f"Skip {folder}: no .png image found.")
                continue
            generated_image_path = gen_images[0]

            metadata_path = os.path.join(folder, "metadata.json")
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            relative_real_path = metadata.get("image_path")
            if not relative_real_path:
                print(f"Skip {metadata_path}: missing 'image_path' field.")
                continue

            real_image_path = os.path.join(real_image_root, relative_real_path)
            if not os.path.exists(real_image_path):
                print(f"Skip sample: real image does not exist: {real_image_path}")
                continue

            folder_name = os.path.basename(folder)
            new_real_image_path = os.path.join(tmp_real_dir, f"{folder_name}_real.png")
            new_generated_image_name = (
                f"{folder_name}_{os.path.basename(generated_image_path)}"
            )
            new_generated_image_path = os.path.join(
                tmp_gen_dir, new_generated_image_name
            )

            resize_and_save(real_image_path, new_real_image_path)
            resize_and_save(generated_image_path, new_generated_image_path)

        except Exception as e:
            print(f"Failed to process folder {folder}: {e}")

    print("\nPrepared sample pairs:")
    print(f"Real images: {len(os.listdir(tmp_real_dir))}")
    print(f"Generated images: {len(os.listdir(tmp_gen_dir))}")

    print("\nComputing FID...")
    try:
        fid_value = fid_score.calculate_fid_given_paths(
            paths=[tmp_real_dir, tmp_gen_dir],
            batch_size=50,
            device=device,
            dims=2048,
        )
        print(f"FID: {fid_value:.2f}")
    except Exception as e:
        print(f"FID computation failed: {e}")

    try:
        calculate_inception_score(
            image_dir=tmp_gen_dir,
            device=device,
            batch_size=50,
        )
    except Exception as e:
        print(f"IS computation failed: {e}")
    finally:
        print("\nCleaning temporary folders...")
        shutil.rmtree(tmp_real_dir)
        shutil.rmtree(tmp_gen_dir)
        print("Temporary folders removed.")


if __name__ == "__main__":
    calculate_fid_and_is()
