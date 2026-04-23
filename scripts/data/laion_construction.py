import io
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

TOTAL_IMAGES = 200000
NUM_THREADS = 80
OUTPUT_DIR = "./datasets/laion"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
METADATA_DIR = os.path.join(OUTPUT_DIR, "metadata")


def setup_directories():
    """Create output folders for downloaded images and per-thread metadata."""
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)


def download_chunk(worker_id, indices, dataset):
    """
    Download a subset of LAION samples for one worker.

    Args:
        worker_id: Integer worker identifier.
        indices: Sample indices assigned to this worker.
        dataset: Preloaded HF dataset subset.
    """
    metadata = []
    print(f"Worker {worker_id} started with {len(indices)} samples.")

    for index in tqdm(indices, desc=f"worker-{worker_id}", position=worker_id):
        try:
            record = dataset[index]
            url = record.get("URL")
            text = record.get("TEXT")

            if not url or not text:
                continue

            response = requests.get(url, timeout=10, stream=True)
            response.raise_for_status()

            content_type = response.headers.get("content-type")
            if content_type and "image" in content_type:
                ext = content_type.split("/")[-1].split(";")[0]
                if ext not in ["jpeg", "png", "gif", "webp"]:
                    ext = "jpg"
            else:
                ext = os.path.splitext(url)[1][1:].lower()
                if not ext:
                    ext = "jpg"

            image_bytes = response.content
            try:
                Image.open(io.BytesIO(image_bytes)).verify()
            except Exception:
                continue

            image_filename = f"{index}.{ext}"
            image_path = os.path.join(IMAGE_DIR, image_filename)
            with open(image_path, "wb") as f:
                f.write(image_bytes)

            metadata.append(
                {
                    "image_path": os.path.join("images", image_filename),
                    "prompt": text,
                }
            )

        except requests.exceptions.RequestException as e:
            print(f"Worker {worker_id}: request failed for sample {index}: {e}")
        except Exception as e:
            print(f"Worker {worker_id}: unexpected failure for sample {index}: {e}")

    if metadata:
        json_path = os.path.join(METADATA_DIR, f"metadata_thread_{worker_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

    print(f"Worker {worker_id} finished. Downloaded {len(metadata)} valid samples.")
    return len(metadata)


def merge_json_files():
    """Merge per-thread metadata files into a single JSON manifest."""
    all_metadata = []
    print("\nMerging metadata JSON files...")

    for filename in os.listdir(METADATA_DIR):
        if filename.startswith("metadata_thread_") and filename.endswith(".json"):
            filepath = os.path.join(METADATA_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_metadata.extend(data)

    final_json_path = os.path.join(OUTPUT_DIR, "final_metadata.json")
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=4)

    print(
        f"Merged metadata saved to {final_json_path} with {len(all_metadata)} entries."
    )


def main():
    """Download LAION subset images and write a unified metadata file."""
    print("Starting LAION subset construction...")

    # Use a mirror endpoint when direct HF access is constrained.
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    setup_directories()

    print("Loading dataset from Hugging Face...")
    ds = load_dataset("dclure/laion-aesthetics-12m-umap")["train"]

    print(f"Taking first {TOTAL_IMAGES} samples...")
    dataset_subset = list(ds.take(TOTAL_IMAGES))

    indices = list(range(TOTAL_IMAGES))
    chunk_size = (TOTAL_IMAGES + NUM_THREADS - 1) // NUM_THREADS
    chunks = [indices[i : i + chunk_size] for i in range(0, TOTAL_IMAGES, chunk_size)]

    print(f"Prepared {len(chunks)} chunks for {NUM_THREADS} workers.")

    total_downloaded = 0
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [
            executor.submit(download_chunk, i, chunks[i], dataset_subset)
            for i in range(len(chunks))
        ]

        for future in as_completed(futures):
            total_downloaded += future.result()

    print(f"\nDownloaded {total_downloaded} / {TOTAL_IMAGES} samples.")

    merge_json_files()
    print("LAION subset construction finished.")


if __name__ == "__main__":
    main()
