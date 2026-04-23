# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import os
import pyarrow.parquet as pq
import random
from PIL import Image

from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset
from .parquet_utils import get_parquet_data_paths, init_arrow_pf_fs

Image.MAX_IMAGE_PIXELS = 20_000_000

use_det_align = False
class T2IIterableDataset(DistributedIterableDataset):
    def __init__(
        self, dataset_name, transform, tokenizer, data_dir_list, num_used_data,
        local_rank=0, world_size=1, num_workers=8, data_status=None,
    ):
        """
        data_dir_list: list of data directories contains parquet files
        num_used_data: list of number of sampled data paths for each data directory
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.data_paths = self.get_data_paths(data_dir_list, num_used_data)

        # response_path = "/path/to/datasets/Unigendet/laion_inference_results/laion_inference_results.json"
        # response_path = "/path/to/datasets/Unigendet/laion_inference_results_prompt/laion_inference_results.json"
        # with open(response_path, 'r', encoding='utf-8') as f:
        # response_data = json.load(f)
        # self.response_map = {item['image_path']: item['response'] for item in response_data if item.get('response', '')}
        with open("/path/to/datasets/Unigendet/laion/final_metadata.json", 'r', encoding='utf-8') as f:
            response_data = json.load(f)
            self.response_map = {item['image_path']: item['response'] for item in response_data if item.get('response', '')}

        self.set_epoch()

    def get_data_paths(self, data_dir_list, num_used_data):
        metadata_path = "/path/to/datasets/Unigendet/laion/final_metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # meta_data_paths = []
        # for data in metadata:
        # meta_data_paths.append(data["image_path"])
        return list(metadata)

    def __iter__(self):
        metadata_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            start_index = self.data_status[worker_id][0] + 1
        else:
            start_index = 0
        transform_stride = self.transform.stride
        image_root = "/path/to/datasets/Unigendet/laion/"

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at index #{start_index}"
        )

        while True:
            metadata_per_worker_ = metadata_per_worker[start_index:]
            for idx, item in enumerate(metadata_per_worker_, start=start_index):
                num_tokens = 0
                try:
                    image_path_key = item['image_path']
                    if use_det_align and image_path_key not in self.response_map:
                        continue
                    image_path = os.path.join(image_root, image_path_key)
                    image = pil_img2rgb(Image.open(image_path))
                except Exception as e:
                    print(f'Error opening image {item["image_path"]}: {e}')
                    continue

                try:
                    image_tensor = self.transform(image)
                    height, width = image_tensor.shape[1:]
                    num_tokens += width * height // transform_stride ** 2
                except Exception as e:
                    print(f'Error transforming image {item["image_path"]}: {e}')
                    continue

                try:
                    caption = item['prompt']
                    # response = self.response_map[image_path_key]
                except Exception as e:
                    print(f'Error getting caption for {item["image_path"]}: {e}')
                    continue

                caption_token = self.tokenizer.encode(caption)
                # response_token = self.tokenizer.encode(response)

                sequence_plan, text_ids_list = [], []
                text_ids = caption_token
                num_tokens += len(caption_token)
                text_ids_list.append(text_ids)
                sequence_plan.append({
                    'type': 'text',
                    'enable_cfg': 1,
                    'loss': 0,
                    'special_token_loss': 0,
                    'special_token_label': None,
                })

                sequence_plan.append({
                    'type': 'vae_image',
                    'enable_cfg': 0,
                    'loss': 1,
                    'special_token_loss': 0,
                    'special_token_label': None,
                })

                sample = dict(
                    gen_image_tensor_list=[image_tensor],
                    text_ids_list=text_ids_list,
                    num_tokens=num_tokens,
                    sequence_plan=sequence_plan,
                    data_indexes={
                        "data_indexes": [idx],
                        "worker_id": worker_id,
                        "dataset_name": self.dataset_name,
                    },
                    generation_auth_explanation_list=["response"],
                    gt_vit_list=[image],
                    gt_prompt_list=[caption],
                )
                yield sample

            start_index = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
