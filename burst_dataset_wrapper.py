"""
Wrapper for DegradationDataset to match BurstM Network.py input format.

Transforms the output of DegradationDataset to the exact tuple format expected by
the training_step in BurstM's Network.py LightningModule.
"""

import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Union


class BurstDatasetWrapper(Dataset):
    """
    Wrapper that transforms DegradationDataset output to BurstM Network.py format.
    
    Network.py training_step expects:
        x, y, flow_vectors, meta_info, downsample_factor, target_size = train_batch
    
    Where:
        - x: Burst of LR images as a list containing tensor [num_frames, C, H, W]
        - y: HR ground truth image [C, H_hr, W_hr]
        - flow_vectors: Optical flow [num_frames, 2, H, W]
        - meta_info: Metadata dictionary
        - downsample_factor: Tensor with single value (downsampling factor)
        - target_size: Tuple (H_target, W_target) for output resolution
    """
    
    def __init__(self, degradation_dataset: Dataset):
        """
        Initialize wrapper.
        
        Args:
            degradation_dataset: Instance of DegradationDataset
        """
        self.dataset = degradation_dataset
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Get transformed sample in BurstM Network.py format.
        
        Returns:
            Tuple of (x, y, flow_vectors, meta_info, downsample_factor, target_size)
        """
        # Get sample from underlying dataset
        sample = self.dataset[idx]
        
        # Extract components
        hr = sample['hr']  # [1, H_hr, W_hr]
        lr_list = sample['lr']  # List of [1, H_lr, W_lr]
        flow_vectors = sample['flow_vectors']  # [num_frames, 2, H_lr, W_lr]
        metadata = sample['metadata']
        
        # Transform LR frames: Stack into single tensor [num_frames, 1, H, W]
        # Network.py expects burst as a list with first element being the stacked tensor
        lr_burst = torch.stack(lr_list, dim=0)  # [num_frames, 1, H_lr, W_lr]
        x = [lr_burst]  # Wrap in list as Network.py expects burst[0]
        
        # Transform HR: Remove batch dimension if needed, keep as [1, H, W] for grayscale
        y = hr  # [1, H_hr, W_hr]
        
        # Flow vectors: Already in correct format [num_frames, 2, H_lr, W_lr]
        # No transformation needed
        
        # Meta info: Pass through the metadata dictionary
        meta_info = metadata
        
        # Downsample factor: Convert to tensor
        downsample_factor = torch.tensor(metadata['downsampling_factor'], dtype=torch.float32)
        
        # Target size: Use half of HR dimensions for output
        # This matches the typical SR setup where output is intermediate resolution
        target_size = (hr.shape[1] // 2, hr.shape[2] // 2)
        
        return x, y, flow_vectors, meta_info, downsample_factor, target_size


def collate_fn(batch: List[Tuple]) -> Tuple:
    """
    Custom collate function for BurstDatasetWrapper.
    
    Batches multiple samples while maintaining the Network.py expected format.
    
    Args:
        batch: List of tuples from __getitem__
        
    Returns:
        Batched tuple (x, y, flow_vectors, meta_info, downsample_factor, target_size)
    """
    # Unpack batch
    x_list, y_list, flow_list, meta_list, ds_factor_list, target_size_list = zip(*batch)
    
    # Batch LR bursts
    # Each x is a list containing [num_frames, 1, H, W]
    # Stack across batch: [B, num_frames, 1, H, W]
    x_burst = torch.stack([x[0] for x in x_list], dim=0)
    x_batched = [x_burst]  # Keep as list for Network.py
    
    # Batch HR images: [B, 1, H_hr, W_hr]
    y_batched = torch.stack(y_list, dim=0)
    
    # Batch flow vectors: [B, num_frames, 2, H, W]
    flow_batched = torch.stack(flow_list, dim=0)
    
    # Meta info: Keep as list of dicts
    meta_info_batched = meta_list
    
    # Downsample factor: Stack into tensor [B]
    downsample_factor_batched = torch.stack(ds_factor_list, dim=0)
    
    # Target size: Use the first sample's target size (should be same for all in batch)
    target_size_batched = target_size_list[0]
    
    return x_batched, y_batched, flow_batched, meta_info_batched, downsample_factor_batched, target_size_batched


class BurstDatasetWrapperV2(Dataset):
    """
    Alternative wrapper with more flexible batching.
    
    Returns individual burst frames in a format that's easier to batch.
    Use this version if the standard wrapper has batching issues.
    """
    
    def __init__(self, degradation_dataset: Dataset):
        self.dataset = degradation_dataset
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get sample as dictionary for easier custom collation.
        
        Returns:
            Dictionary with all components
        """
        sample = self.dataset[idx]
        
        # Stack LR frames
        lr_burst = torch.stack(sample['lr'], dim=0)  # [num_frames, 1, H, W]
        
        return {
            'burst': lr_burst,  # [num_frames, 1, H, W]
            'hr': sample['hr'],  # [1, H_hr, W_hr]
            'flow_vectors': sample['flow_vectors'],  # [num_frames, 2, H, W]
            'metadata': sample['metadata'],
            'downsample_factor': sample['metadata']['downsampling_factor'],
        }


def collate_fn_v2(batch: List[Dict]) -> Tuple:
    """
    Collate function for BurstDatasetWrapperV2.
    
    Args:
        batch: List of dictionaries from __getitem__
        
    Returns:
        Tuple matching Network.py format
    """
    # Stack bursts: [B, num_frames, 1, H, W]
    bursts = torch.stack([sample['burst'] for sample in batch], dim=0)
    x = [bursts]
    
    # Stack HR: [B, 1, H_hr, W_hr]
    y = torch.stack([sample['hr'] for sample in batch], dim=0)
    
    # Stack flows: [B, num_frames, 2, H, W]
    flows = torch.stack([sample['flow_vectors'] for sample in batch], dim=0)
    
    # Metadata list
    meta_info = [sample['metadata'] for sample in batch]
    
    # Downsample factors: [B]
    ds_factors = torch.tensor([sample['downsample_factor'] for sample in batch], dtype=torch.float32)
    
    # Target size from first sample
    hr_shape = batch[0]['hr'].shape
    target_size = (hr_shape[1] // 2, hr_shape[2] // 2)
    
    return x, y, flows, meta_info, ds_factors, target_size
