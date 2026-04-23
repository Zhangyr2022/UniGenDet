import io
import json
import os
import traceback
import random
from PIL import Image, ImageFile, PngImagePlugin

from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset
from .transforms import ImageTransform

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte
USE_GENETATION_VAE_IMAGE_TOKEN=True

class DetectionSftJSONLIterableDataset(DistributedIterableDataset):
    def __init__(
        self, dataset_name, transform, tokenizer, frame_sampler,
        data_dir_list,num_used_data,
        json_path="./datasets/fakeclue/data_json/train.json", data_root="./datasets/fakeclue/train", local_rank=0, world_size=1, num_workers=8, data_status=None,
        shuffle_lines=False,shuffle_data=True, shuffle_seed=0,
    ):
        """
        json_path: path to the json file which contains a list of data annotations.
        data_root: The root directory for the image paths in the json file.
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.frame_sampler = frame_sampler
        self.data_status = data_status
        self.data_root = data_root
        self.shuffle_data = shuffle_data
        self.shuffle_seed = shuffle_seed
        self.vae_transform = ImageTransform(
            max_image_size =1024,
            min_image_size=512,
            image_stride=16,
        )
        with open(json_path, 'r') as f:
            self.all_data = json.load(f)
        # all_data ['label'] 01
        label_0_count = sum(1 for item in self.all_data if item.get('label') == 0)
        label_1_count = sum(1 for item in self.all_data if item.get('label') == 1)
        min_count = min(label_0_count, label_1_count)
        balanced_data = []
        label_0_added = 0
        label_1_added = 0
        random.seed(42)
        random.shuffle(self.all_data)

        for item in self.all_data:
            label = item.get('label')
            if label == 0 and label_0_added < min_count:
                balanced_data.append(item)
                label_0_added += 1
            elif label == 1 and label_1_added < min_count:
                balanced_data.append(item)
                label_1_added += 1
        print(f"Balanced dataset from {len(self.all_data)} items to 2*{min_count} samples for each label.")
        self.all_data = balanced_data
        self.set_epoch()

    def __iter__(self):
        # Shuffle data at the beginning of each epoch if requested
        if self.shuffle_data:
            g = random.Random(self.shuffle_seed)
            g.shuffle(self.all_data)

        # Distribute data among workers
        total_size = len(self.all_data)
        items_per_worker = total_size // (self.world_size * self.num_workers)
        worker_id = self.local_rank * self.num_workers
        start_index = worker_id * items_per_worker
        end_index = start_index + items_per_worker

        # The last worker in the world gets the remainder
        if worker_id == self.world_size * self.num_workers - 1:
            end_index = total_size

        data_per_worker = self.all_data[start_index:end_index]

        # Resume from a specific index if data_status is provided
        start_row = 0
        if self.data_status is not None and self.data_status.get(worker_id):
            start_row = self.data_status[worker_id][0] + 1

        transform_stride = self.transform.stride

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"processing {len(data_per_worker)} items. Resuming from item #{start_row}"
        )

        while True:
            for row_idx, row in enumerate(data_per_worker[start_row:]):
                current_row_idx = start_row + row_idx
                num_tokens = 0
                vit_image_tensor_list = []
                gen_image_tensor_list = []
                text_ids_list = []
                sequence_plan = []

                try:
                    image_path = os.path.join(self.data_root, row['image'])
                    image = pil_img2rgb(Image.open(image_path))
                except Exception as e:
                    print(f'Error opening image {row.get("image")}: {e}')
                    continue
                vae_image_tensor = self.vae_transform(image)
                gen_image_tensor_list.append(vae_image_tensor)

                image_tensor = self.transform(image)
                vit_image_tensor_list.append(image_tensor)


                height, width = image_tensor.shape[1:]
                num_tokens += width * height // transform_stride ** 2

                vae_height, vae_width = vae_image_tensor.shape[1:]
                num_tokens += vae_width * vae_height // self.vae_transform.stride ** 2

                conversations = row.get('conversations', [])
                if not conversations:
                    continue

                # <image> uid
                has_image_token = False
                for item in conversations:
                    if item['from'] == 'human':
                        text_data = item['value']
                        if '<image>' in text_data and not has_image_token:
                            parts = text_data.split('<image>')
                            if parts[0]:
                                text_ids = self.tokenizer.encode(parts[0])
                                text_ids_list.append(text_ids)
                                num_tokens += len(text_ids)
                                sequence_plan.append({'type': 'text', 'loss': 0})

                            if USE_GENETATION_VAE_IMAGE_TOKEN:
                                sequence_plan.append({'type': 'vae_image', 'loss': 0})
                            sequence_plan.append({'type': 'vit_image', 'loss': 0})
                            has_image_token = True

                            if parts[1]:
                                text_ids = self.tokenizer.encode(parts[1])
                                text_ids_list.append(text_ids)
                                num_tokens += len(text_ids)
                                sequence_plan.append({'type': 'text', 'loss': 0})

                        else:
                            text_ids = self.tokenizer.encode(text_data)
                            text_ids_list.append(text_ids)
                            num_tokens += len(text_ids)
                            sequence_plan.append({'type': 'text', 'loss': 0})
                    elif item['from'] == 'gpt':
                        text_data = item['value']
                        text_ids = self.tokenizer.encode(text_data)
                        text_ids_list.append(text_ids)
                        num_tokens += len(text_ids)
                        sequence_plan.append({'type': 'text', 'loss': 1})

                # Finalize sequence plan details
                for plan in sequence_plan:
                    plan.update({
                        'enable_cfg': 0,
                        'special_token_loss': 0,
                        'special_token_label': None,
                    })

                has_loss = [item['loss'] for item in sequence_plan]
                if sum(has_loss) == 0:
                    print(f'No loss defined for item {current_row_idx}, skipped.')
                    continue

                yield dict(
                    vit_image_tensor_list=vit_image_tensor_list,
                    gen_image_tensor_list=gen_image_tensor_list,
                    text_ids_list=text_ids_list,
                    sequence_plan=sequence_plan,
                    num_tokens=num_tokens,
                    data_indexes={
                        "data_indexes": [current_row_idx],
                        "worker_id": worker_id,
                        "dataset_name": self.dataset_name,
                    },
                    label=int(row.get('label', -1))
                )

            start_row = 0 # Reset for the next epoch
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
            # Break the loop to avoid infinite repetition within a single `__iter__` call
            # The training loop is expected to create a new iterator for each epoch.
            # break