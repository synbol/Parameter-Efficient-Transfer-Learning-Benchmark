#!/usr/bin/env python3

"""Data loader."""
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

# from ..utils import logging
from .json_dataset import (
    CUB200Dataset, CarsDataset, DogsDataset, FlowersDataset, NabirdsDataset
)

# logger = logging.get_logger("visual_prompt")
_DATASET_CATALOG = {
    "CUB_200_2011": CUB200Dataset,
    'OxfordFlower': FlowersDataset,
    'StanfordCars': CarsDataset,
    'StanfordDogs': DogsDataset,
    "nabirds": NabirdsDataset,
}


def _construct_loader(dataset_name, split, batch_size, shuffle, drop_last):
    """Constructs the data loader for the given dataset."""

    # Construct the dataset
    if dataset_name.startswith("vtab-"):
        # import the tensorflow here only if needed
        from .datasets.tf_dataset import TFDataset
        dataset = TFDataset(cfg, split)
    else:
        assert (
            dataset_name in _DATASET_CATALOG.keys()
        ), "Dataset '{}' not supported".format(dataset_name)
        dataset = _DATASET_CATALOG[dataset_name](dataset_name, split)

    # Create a sampler for multi-process training
    # sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # num_workers=cfg.DATA.NUM_WORKERS,
        # pin_memory=cfg.DATA.PIN_MEMORY,
        drop_last=drop_last,
    )
    return loader


def construct_train_loader(dataset_name, batch_size=None):
    """Train loader wrapper."""
    drop_last = False
    return _construct_loader(
        dataset_name=dataset_name,
        split="train",
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
    )


def construct_trainval_loader(dataset_name, batch_size=None):
    if batch_size is None:
        bs = int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS)
    else:
        bs = batch_size
    drop_last = False
    return _construct_loader(
        dataset_name=dataset_name,
        split="trainval",
        batch_size=bs,
        shuffle=True,
        drop_last=drop_last,
    )


def construct_test_loader(dataset_name, batch_size=None):
    if batch_size is None:
        bs = int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS)
    else:
        bs = batch_size
    """Test loader wrapper."""
    return _construct_loader(
        dataset_name=dataset_name,
        split="test",
        batch_size=bs,
        shuffle=False,
        drop_last=False,
    )


def construct_val_loader(dataset_name, batch_size=None):
    if batch_size is None:
        bs = int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS)
    else:
        bs = batch_size
    """Validation loader wrapper."""
    return _construct_loader(
        dataset_name=dataset_name,
        split="val",
        batch_size=bs,
        shuffle=False,
        drop_last=False,
    )


def shuffle(loader, cur_epoch):
    """"Shuffles the data."""
    assert isinstance(
        loader.sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(loader.sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)