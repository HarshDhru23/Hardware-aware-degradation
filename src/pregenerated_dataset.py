"""
PyTorch Dataset class for loading pre-generated training data.

This dataset loads samples that were generated and saved by generate_training_dataset.py.
It provides the same interface as DegradationDataset but loads from disk instead of
generating on-the-fly.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import yaml
import numpy as np
import json
import logging


class PreGeneratedDataset(Dataset):
    """
    PyTorch Dataset for loading pre-generated degraded images.
    
    This dataset loads samples that were pre-generated using generate_training_dataset.py,
    providing fast loading without on-the-fly degradation computation.
    """
    
    def __init__(
        self,
        dataset_dir: Union[str, Path],
        transform: Optional[Any] = None,
        load_flow: bool = True,
        load_psf: bool = True
    ):
        """
        Initialize PreGeneratedDataset.
        
        Args:
            dataset_dir: Directory containing the pre-generated dataset
            transform: Optional transform to apply to samples
            load_flow: Whether to load flow vectors (set False to save memory)
            load_psf: Whether to load PSF kernels (set False to save memory)
        """
        super().__init__()
        
        self.dataset_dir = Path(dataset_dir)
        self.samples_dir = self.dataset_dir / "samples"
        self.transform = transform
        self.load_flow = load_flow
        self.load_psf = load_psf
        
        self.logger = logging.getLogger(__name__)
        
        # Load dataset info
        dataset_info_path = self.dataset_dir / 'dataset_info.yaml'
        if not dataset_info_path.exists():
            raise FileNotFoundError(f"Dataset info not found: {dataset_info_path}")
        
        with open(dataset_info_path, 'r') as f:
            self.dataset_info = yaml.safe_load(f)
        
        self.format = self.dataset_info['format']
        self.total_samples = self.dataset_info['total_samples']
        self.num_lr_frames = self.dataset_info['num_lr_frames']
        
        # Validate samples directory
        if not self.samples_dir.exists():
            raise FileNotFoundError(f"Samples directory not found: {self.samples_dir}")
        
        # Build sample paths
        file_ext = '.npz' if self.format == 'npz' else '.pt'
        self.sample_paths = [
            self.samples_dir / f"sample_{idx:06d}{file_ext}"
            for idx in range(self.total_samples)
        ]
        
        # Verify at least first and last sample exist
        if not self.sample_paths[0].exists():
            raise FileNotFoundError(f"First sample not found: {self.sample_paths[0]}")
        if not self.sample_paths[-1].exists():
            raise FileNotFoundError(f"Last sample not found: {self.sample_paths[-1]}")
        
        self.logger.info(f"PreGeneratedDataset initialized:")
        self.logger.info(f"  Dataset directory: {self.dataset_dir}")
        self.logger.info(f"  Total samples: {self.total_samples}")
        self.logger.info(f"  Format: {self.format}")
        self.logger.info(f"  LR frames per sample: {self.num_lr_frames}")
        self.logger.info(f"  Load flow vectors: {self.load_flow}")
        self.logger.info(f"  Load PSF kernels: {self.load_psf}")
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                'hr': HR image tensor [1, H, W]
                'lr': List of LR image tensors [1, H/scale, W/scale]
                'flow_vectors': Flow vectors tensor [num_frames, 2, H, W] (optional)
                'psf_kernels': List of PSF kernel tensors [Kh, Kw] (optional)
                'psf_params': Dict with 'sigma_x', 'sigma_y', 'theta' lists
                'shift_values': List of shift values [(sx, sy), ...]
                'metadata': Dict with filename, augmentation, etc.
        """
        sample_path = self.sample_paths[idx]
        
        if not sample_path.exists():
            raise FileNotFoundError(f"Sample not found: {sample_path}")
        
        try:
            if self.format == 'npz':
                return self._load_npz(sample_path)
            elif self.format == 'pt':
                return self._load_pt(sample_path)
            else:
                raise ValueError(f"Unsupported format: {self.format}")
        
        except Exception as e:
            self.logger.error(f"Failed to load sample {idx} from {sample_path}: {e}")
            raise
    
    def _load_npz(self, path: Path) -> Dict[str, Any]:
        """Load sample from NPZ format."""
        data = np.load(path, allow_pickle=True)
        
        # Load HR and LR images
        hr = torch.from_numpy(data['hr']).float()
        lr_frames_np = data['lr_frames']
        lr_frames = [torch.from_numpy(lr).float() for lr in lr_frames_np]
        
        # Load shift values
        shift_values = data['shift_values'].tolist()
        
        # Load PSF parameters
        psf_params = {
            'sigma_x': data['psf_sigma_x'].tolist(),
            'sigma_y': data['psf_sigma_y'].tolist(),
            'theta': data['psf_theta'].tolist()
        }
        
        # Load metadata
        metadata = json.loads(str(data['metadata']))
        
        # Prepare output
        output = {
            'hr': hr,
            'lr': lr_frames,
            'psf_params': psf_params,
            'shift_values': shift_values,
            'metadata': metadata
        }
        
        # Load flow vectors if requested
        if self.load_flow and 'flow_vectors' in data:
            output['flow_vectors'] = torch.from_numpy(data['flow_vectors']).float()
        
        # Load PSF kernels if requested
        if self.load_psf:
            psf_kernels = []
            for i in range(self.num_lr_frames):
                key = f'psf_kernel_{i}'
                if key in data:
                    psf_kernels.append(torch.from_numpy(data[key]).float())
                else:
                    psf_kernels.append(None)
            output['psf_kernels'] = psf_kernels
        
        return output
    
    def _load_pt(self, path: Path) -> Dict[str, Any]:
        """Load sample from PyTorch format."""
        data = torch.load(path)
        
        # Optionally remove flow vectors or PSF kernels to save memory
        if not self.load_flow and 'flow_vectors' in data:
            del data['flow_vectors']
        
        if not self.load_psf and 'psf_kernels' in data:
            del data['psf_kernels']
        
        return data
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        return self.dataset_info.copy()
    
    def get_sample_metadata(self, idx: int) -> Dict[str, Any]:
        """
        Get metadata for a specific sample without loading the full sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Metadata dictionary
        """
        sample = self[idx]
        return sample['metadata']


def create_pregenerated_dataloader(
    dataset_dir: Union[str, Path],
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    transform: Optional[Any] = None,
    load_flow: bool = True,
    load_psf: bool = True
) -> DataLoader:
    """
    Create a DataLoader for pre-generated dataset.
    
    Args:
        dataset_dir: Directory containing pre-generated dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        transform: Optional transform
        load_flow: Whether to load flow vectors
        load_psf: Whether to load PSF kernels
        
    Returns:
        DataLoader instance
    """
    dataset = PreGeneratedDataset(
        dataset_dir=dataset_dir,
        transform=transform,
        load_flow=load_flow,
        load_psf=load_psf
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn_pregenerated
    )
    
    return dataloader


def collate_fn_pregenerated(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """
    Custom collate function for PreGeneratedDataset.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Batched dictionary
    """
    # Stack HR images
    hr_batch = torch.stack([sample['hr'] for sample in batch], dim=0)
    
    # Stack LR frames (each sample has a list of LR frames)
    num_lr_frames = len(batch[0]['lr'])
    lr_batch = []
    for i in range(num_lr_frames):
        lr_frame_batch = torch.stack([sample['lr'][i] for sample in batch], dim=0)
        lr_batch.append(lr_frame_batch)
    
    # Collect shift values
    shift_values_batch = [sample['shift_values'] for sample in batch]
    
    # Collect PSF parameters
    psf_params_batch = {
        'sigma_x': [sample['psf_params']['sigma_x'] for sample in batch],
        'sigma_y': [sample['psf_params']['sigma_y'] for sample in batch],
        'theta': [sample['psf_params']['theta'] for sample in batch]
    }
    
    # Collect metadata
    metadata_batch = [sample['metadata'] for sample in batch]
    
    output = {
        'hr': hr_batch,
        'lr': lr_batch,
        'shift_values': shift_values_batch,
        'psf_params': psf_params_batch,
        'metadata': metadata_batch
    }
    
    # Stack flow vectors if present
    if 'flow_vectors' in batch[0]:
        flow_batch = torch.stack([sample['flow_vectors'] for sample in batch], dim=0)
        output['flow_vectors'] = flow_batch
    
    # Collect PSF kernels if present
    if 'psf_kernels' in batch[0]:
        psf_kernels_batch = []
        for i in range(num_lr_frames):
            kernels = [sample['psf_kernels'][i] for sample in batch if sample['psf_kernels'][i] is not None]
            if kernels:
                psf_kernels_batch.append(torch.stack(kernels, dim=0))
            else:
                psf_kernels_batch.append(None)
        output['psf_kernels'] = psf_kernels_batch
    
    return output


# Example usage
if __name__ == "__main__":
    """Example of how to use PreGeneratedDataset."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load dataset
    dataset = PreGeneratedDataset(
        dataset_dir='data/training_dataset',
        load_flow=True,
        load_psf=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset info: {dataset.get_dataset_info()}")
    
    # Get a sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  HR shape: {sample['hr'].shape}")
    print(f"  LR frames: {len(sample['lr'])}")
    print(f"  LR[0] shape: {sample['lr'][0].shape}")
    print(f"  Metadata: {sample['metadata']}")
    
    # Create dataloader
    dataloader = create_pregenerated_dataloader(
        dataset_dir='data/training_dataset',
        batch_size=4,
        shuffle=True,
        num_workers=2
    )
    
    # Iterate through batches
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"  HR batch shape: {batch['hr'].shape}")
        print(f"  LR batch frames: {len(batch['lr'])}")
        print(f"  LR[0] batch shape: {batch['lr'][0].shape}")
        
        if batch_idx >= 2:  # Just show first 3 batches
            break
