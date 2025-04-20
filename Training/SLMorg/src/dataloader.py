import torch
import torch.distributed as dist
import os
import warnings

from typing import Union, List
from torch.utils.data import Dataset, DataLoader, DistributedSampler, SequentialSampler

class TinyStoriesDataset(Dataset):
    def __init__(self, X, y):
        """
        Initialize the dataset with in-memory tensors.

        Args:
            X: Tensor containing the input data
            y: Tensor containing the corresponding labels
        """
        self.X = X
        self.y = y
        self.total_len = len(X)

    def __len__(self):
        """Return the total length of the dataset."""
        return self.total_len

    def __getitem__(self, idx):
        """
        Fetch an item by index.

        Returns:
            Tuple of (X[idx], y[idx])
        """
        return self.X[idx], self.y[idx]

def get_data(path):
    """
    Load all data from the specified directory into memory as tensors.

    Args:
        path: Directory containing 'X' and 'y' subdirectories with data files

    Returns:
        Tuple of (X, y), where X and y are concatenated tensors from all files

    Raises:
        OSError: If no data files are found in the directories
        ValueError: If the total number of samples in X and y do not match
    """
    X_dir = os.path.join(path, 'X')
    y_dir = os.path.join(path, 'y')

    X_pths = sorted([os.path.join(X_dir, f) for f in os.listdir(X_dir)])
    y_pths = sorted([os.path.join(y_dir, f) for f in os.listdir(y_dir)])

    if not X_pths or not y_pths:
        raise OSError("No data files found in the specified directories.")

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        X_list = [torch.load(pth, map_location='cpu') for pth in X_pths]
        y_list = [torch.load(pth, map_location='cpu') for pth in y_pths]

    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)

    if X.size(0) != y.size(0):
        raise ValueError("Total number of samples in X and y do not match.")

    print('Successfully loaded data as tensors')
    return X, y

def get_dataloader(
    X,
    y,
    batch_size,
    num_workers,
    shuffle: bool = False,
    pin_memory: bool = False,
    parallelism_type: str = None,
    rank: int = None,
):
    """
    Create a DataLoader for the dataset, with support for distributed training.

    Args:
        X: Tensor containing the input data
        y: Tensor containing the corresponding labels
        batch_size: Number of samples per batch
        num_workers: Number of subprocesses for data loading
        shuffle: Whether to shuffle the data (handled by sampler in distributed mode)
        pin_memory: Whether to pin memory for faster GPU transfer
        parallelism_type: 'fsdp', 'ddp', 'dp', or None (determines sampler type)
        rank: Process rank (required for FSDP/DDP)

    Returns:
        DataLoader configured for the specified parallelism type
    """
    print("Loading dataset")
    dataset = TinyStoriesDataset(X, y)
    _val_distributed = ['fsdp', 'ddp']

    if parallelism_type in _val_distributed:
        if rank is None:
            raise ValueError("Rank must be provided for DDP/FSDP")
        print(f"Loading Distributed Sampler for {parallelism_type}")
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=shuffle
        )
    else:
        print(f"Loading sequential sampler{' for DP' if parallelism_type == 'dp' else ''}")
        sampler = SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return dataloader