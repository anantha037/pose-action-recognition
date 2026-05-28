"""
train/dataset.py

PyTorch Dataset class for loading .npy keypoint sequences, preprocessing them, 
and preparing them for LSTM training.
"""

import os
import glob
import random
import copy
from typing import List, Tuple, Dict, Optional, Callable, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler


def compute_dataset_stats(samples: List[Tuple[str, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes the full dataset sample paths, iterates all samples, and computes global
    mean and std per feature across all sequences.
    
    Note: Computing stats on the full dataset before splitting introduces slight data
    leakage, but this is acceptable for small datasets to ensure stable normalization.
    
    Args:
        samples (List[Tuple[str, int]]): List of tuples containing (file_path, label_index).
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Mean and std arrays of shape (132,).
    """
    all_features = []
    for file_path, _ in samples:
        seq = np.load(file_path)
        all_features.append(seq)
        
    if not all_features:
        return np.zeros(132, dtype=np.float32), np.ones(132, dtype=np.float32)
        
    # Concatenate all frames across all sequences
    all_features_np = np.concatenate(all_features, axis=0)
    
    mean = np.mean(all_features_np, axis=0)
    std = np.std(all_features_np, axis=0)
    return mean, std


class ActionSequenceDataset(Dataset):
    """
    PyTorch Dataset for loading MediaPipe keypoint sequences from .npy files.
    """
    
    def __init__(
        self,
        data_dir: str = os.path.join("data", "sequences"),
        actions: Optional[List[str]] = None,
        sequence_length: int = 30,
        transform: Optional[Callable] = None,
        augment: bool = False
    ):
        """
        Initializes the dataset.
        
        Args:
            data_dir (str): Path to the root directory containing sequence files.
            actions (Optional[List[str]]): List of action class names. Defaults to standard 5 classes.
            sequence_length (int): Expected frames per sequence. Shorter are padded, longer are cropped.
            transform (Optional[Callable]): Optional transform applied to the sequence tensor.
            augment (bool): Whether to apply data augmentation (noise, masking, flipping).
        """
        if actions is None:
            actions = ['idle', 'wave', 'squat', 'push_up', 'jumping_jack']
            
        self.data_dir = data_dir
        self.actions = sorted(actions)  # Always sort alphabetically for reproducibility
        self.sequence_length = sequence_length
        self.transform = transform
        self.augment = augment
        
        self.label_map = {action: i for i, action in enumerate(self.actions)}
        self.samples: List[Tuple[str, int]] = []
        
        # Scan data_dir
        class_counts = {action: 0 for action in self.actions}
        for action in self.actions:
            action_dir = os.path.join(self.data_dir, action)
            if os.path.isdir(action_dir):
                # Standardize path slashes for cross-platform compatibility
                files = glob.glob(os.path.join(action_dir, "*.npy"))
                class_counts[action] = len(files)
                for f in files:
                    self.samples.append((f, self.label_map[action]))
                    
        total_samples = len(self.samples)
        
        # Class weights for weighted sampling
        num_classes = len(self.actions)
        self.class_weights = torch.zeros(num_classes)
        for action, idx in self.label_map.items():
            count = class_counts[action]
            if count > 0:
                self.class_weights[idx] = total_samples / (num_classes * count)
                
        # Compute global stats
        self.mean, self.std = compute_dataset_stats(self.samples)
            
        # Print dataset summary
        print(f"Dataset loaded from {self.data_dir}")
        for action in self.actions:
            print(f"  {action:<14}: {class_counts[action]} sequences")
        print("  " + "─" * 29)
        print(f"  Total         : {total_samples} sequences")
        print(f"  Label map     : {self.label_map}")

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Loads, preprocesses, and optionally augments a single sequence.
        
        Args:
            idx (int): Index of the sample to fetch.
            
        Returns:
            Tuple[torch.Tensor, int]: Preprocessed sequence tensor and its label index.
        """
        file_path, label_index = self.samples[idx]
        seq = np.load(file_path)  # Shape: (frames, 132)
        
        # Handle sequences shorter than sequence_length
        if len(seq) < self.sequence_length:
            pad_amount = self.sequence_length - len(seq)
            # Zero-padding at the end
            seq = np.pad(seq, ((0, pad_amount), (0, 0)), mode='constant', constant_values=0)
        # Handle sequences longer than sequence_length
        elif len(seq) > self.sequence_length:
            start = (len(seq) - self.sequence_length) // 2
            seq = seq[start : start + self.sequence_length]
            
        # Normalize (per-feature z-score normalization)
        seq = (seq - self.mean) / (self.std + 1e-8)
        
        # Apply data augmentations
        if self.augment:
            # Gaussian noise (p=0.5)
            if random.random() < 0.5:
                noise = np.random.normal(0, 0.01, seq.shape)
                # Apply noise to x, y, z channels only (every column EXCEPT every 4th starting at 3)
                mask = np.ones(132, dtype=bool)
                mask[3::4] = False
                seq[:, mask] += noise[:, mask]
                
            # Time masking (p=0.3)
            if random.random() < 0.3:
                mask_len = random.randint(1, 3)
                start = random.randint(0, self.sequence_length - mask_len)
                seq[start : start + mask_len, :] = 0
                
            # Horizontal flip (p=0.5)
            if random.random() < 0.5:
                # Negate x-coordinates (every 4th column starting at 0)
                seq[:, 0::4] *= -1
                
        if self.transform:
            seq = self.transform(seq)
            
        return torch.FloatTensor(seq), label_index


def get_data_loaders(
    data_dir: str = os.path.join("data", "sequences"),
    actions: Optional[List[str]] = None,
    batch_size: int = 32,
    val_split: float = 0.2,
    test_split: float = 0.1,
    num_workers: int = 2,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Splits the dataset and returns DataLoaders for training, validation, and testing.
    
    Args:
        data_dir (str): Path to sequence files.
        actions (Optional[List[str]]): List of action class names.
        batch_size (int): Batch size for DataLoaders.
        val_split (float): Fraction of data for validation.
        test_split (float): Fraction of data for testing.
        num_workers (int): Number of subprocesses for data loading.
        seed (int): Random seed for reproducibility.
        
    Returns:
        Dict[str, Any]: Dictionary containing DataLoaders and dataset metadata.
    """
    # 1. Load full dataset (computes global stats internally)
    full_dataset = ActionSequenceDataset(data_dir=data_dir, actions=actions, augment=False)
    
    total_len = len(full_dataset)
    if total_len == 0:
        raise ValueError(f"No data found in {data_dir}. Cannot create loaders.")
        
    val_len = int(val_split * total_len)
    test_len = int(test_split * total_len)
    train_len = total_len - val_len - test_len
    
    # Split using fixed seed
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_len, val_len, test_len], generator=generator
    )
    
    # 2. Create specific dataset instances for each split via shallow copy
    train_dataset = copy.copy(full_dataset)
    train_dataset.samples = [full_dataset.samples[i] for i in train_subset.indices]
    train_dataset.augment = True  # Training set uses augment=True
    
    val_dataset = copy.copy(full_dataset)
    val_dataset.samples = [full_dataset.samples[i] for i in val_subset.indices]
    
    test_dataset = copy.copy(full_dataset)
    test_dataset.samples = [full_dataset.samples[i] for i in test_subset.indices]
    
    # 3. Create WeightedRandomSampler for training set
    train_weights = [train_dataset.class_weights[label] for _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights), replacement=True)
    
    # 4. Initialize DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'label_map': full_dataset.label_map,
        'class_weights': full_dataset.class_weights,
        'mean': full_dataset.mean,
        'std': full_dataset.std
    }


if __name__ == "__main__":
    # Smoke test: instantiate dataloaders and print batch shapes
    print("Initializing DataLoaders (this will print dataset summary)...")
    try:
        # Use 0 workers on Windows for a simple smoke test
        loaders = get_data_loaders(num_workers=0)
        sequences, labels = next(iter(loaders['train']))
        
        print("\n--- Smoke Test ---")
        print(f'Batch shape: {sequences.shape}')  # Expected: (32, 30, 132)
        print(f'Labels shape: {labels.shape}')    # Expected: (32,)
    except ValueError as e:
        print(f"Smoke test failed: {e}")
