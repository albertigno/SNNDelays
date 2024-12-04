import os

CHECKPOINT_PATH = os.path.join(
    os.environ.get('SNN_CHECKPOINTS_PATH'))

# CHECKPOINT_PATH = os.path.join(
#     os.environ.get('SNN_DATA_PATH'), 'Checkpoints')

DATASET_PATH = os.path.join(
    os.environ.get('SNN_DATASETS_PATH'), 'Datasets')

# DATASET_PATH = "E:\SNN_DATASETS\Datasets"

CACHE_PATH = os.path.join(os.environ.get('SNN_DATASETS_PATH'), 'DiskCachedDatasets')

# CACHE_PATH = "E:\SNN_DATASETS\DiskCachedDatasets"