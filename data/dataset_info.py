from .t2i_dataset import T2IIterableDataset
from .detection_json_dataset import DetectionSftJSONLIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': DetectionSftJSONLIterableDataset,
}


DATASET_INFO = {
    't2i_pretrain': {
        't2i': {
            'data_dir': '/path/to/LAION-dataset', # path of the parquet files
            'num_files': 1000, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 80000, # number of total samples in the dataset
        },
    },
    'vlm_sft': {
        'llava_ov': {
            'data_dir': '/path/to/FakeVLM', # path of the parquet files
            'num_files': 1000, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 100000, # number of total samples in the dataset
        },
    },
}