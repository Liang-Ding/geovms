import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import os
import h5py

feature_key       = "X"
label_key         = "y"
threshold         = 0.85    # Baseline, control halo size

def create_dataloader(data_path, batch_size, num_workers, distributed=False, rank=0, world_size=1):
    r"""Create data loader"""
    hdf5_file_paths = []
    for dirpath, _, filenames in os.walk(data_path):
        for filename in filenames:
            if filename.lower().endswith(('.hdf5', '.h5')):
                hdf5_file_paths.append(os.path.join(dirpath, filename))

    dataset = Scraper_h5(hdf5_file_paths)
    # Create samplers for multiple GPUs (distributed=True) or single GPU (distributed=False)
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle = False  # Sampler handles shuffling
    else:
        sampler = None
        shuffle = False  # or True if you prefer shuffling on single-GPU

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True,
    )
    return dataloader


class Scraper_h5(Dataset):
    def __init__(self, hdf5_file_paths):
        self.hdf5_file_paths = hdf5_file_paths
        self.file_handles = [None] * len(hdf5_file_paths)  # Lazy load files
        self.lengths = []
        self.total_length = 0

        for path in self.hdf5_file_paths:
            with h5py.File(path, 'r') as f:
                self.lengths.append(f[feature_key].shape[0])
                self.total_length += f[feature_key].shape[0]

    def __len__(self):
        return self.total_length

    def get_file_and_index(self, idx):
        # Identify which file the index belongs to and the local index within that file
        for i, length in enumerate(self.lengths):
            if idx < length:
                return i, idx
            idx -= length
        raise IndexError('Index out of range')

    def __getitem__(self, idx):
        file_idx, local_idx = self.get_file_and_index(idx)
        # Open the HDF5 file if it hasn't been opened
        if self.file_handles[file_idx] is None:
            self.file_handles[file_idx] = h5py.File(self.hdf5_file_paths[file_idx], 'r')

        features = self.file_handles[file_idx][feature_key][local_idx]
        label = self.file_handles[file_idx][label_key][local_idx]

        # Feature: (C, H, W) [total:85, Aeromagnetic(0-21), Gravity(22-47), EM(48-50), Geology(51-85)]
        features = torch.tensor(features, dtype=torch.float32).contiguous()

        # Label
        label = torch.tensor(label, dtype=torch.float32)
        label = torch.where(label < threshold, torch.tensor(0.0), torch.tensor(1.0))

        return features, label
