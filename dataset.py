import torch
import numpy as np
from torch.utils.data import Dataset

class SkeletalDataset(Dataset):
    """
    PyTorch Dataset for ST-GCN skeleton sequences.
    Loads data and applies real-time data augmentations for training.
    """
    def __init__(self, x_path, y_path, augment=False, rotation_range=15, scale_range=0.1, noise_std=0.01, joint_dropout=0.1):
        """
        x_path: path to features file (.npy) of shape (N, 3, T, V, 1)
        y_path: path to labels file (.npy) of shape (N,)
        augment: Whether to apply augmentations (True for training set)
        """
        self.X = np.load(x_path)
        self.y = np.load(y_path)
        self.augment = augment
        
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.joint_dropout = joint_dropout
        
        assert len(self.X) == len(self.y), f"Features and labels count mismatch: {len(self.X)} vs {len(self.y)}"

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # x shape: (3, T, V, 1)
        x = self.X[idx].copy()
        y = self.y[idx]
        
        if self.augment:
            x = self._apply_augmentation(x)
            
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def _apply_augmentation(self, x):
        # x is (3, T, V, 1) where channel 0: x, channel 1: y, channel 2: confidence
        # Rotate, scale and add noise only to coords (first 2 channels)
        coords = x[:2, :, :, :]  # (2, T, V, 1)
        scores = x[2:3, :, :, :]  # (1, T, V, 1)
        
        # We only augment joints that are valid (confidence > 0)
        valid_mask = scores > 0.1  # (1, T, V, 1)
        
        # 1. Random 2D Rotation around origin
        if self.rotation_range > 0:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            rad = np.radians(angle)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            # Reshape coords for multiplication
            # coords is (2, T, V, 1) -> transpose to (T, V, 2)
            c_trans = coords.squeeze(-1).transpose(1, 2, 0)
            c_rotated = np.einsum('tvj,ij->tvi', c_trans, rot_matrix)
            coords = c_rotated.transpose(2, 0, 1)[:, :, :, np.newaxis]
            
        # 2. Random Scaling
        if self.scale_range > 0:
            scale = np.random.uniform(1.0 - self.scale_range, 1.0 + self.scale_range)
            coords = coords * scale
            
        # 3. Random Gaussian Jitter
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, size=coords.shape)
            # Apply noise only to valid points
            coords = coords + (noise * valid_mask)
            
        # 4. Joint Masking (Dropout)
        # Randomly zero out joint coords to simulate occlusion/loss of tracking
        if self.joint_dropout > 0:
            num_joints = x.shape[2]
            # Choose a random set of joints to drop for this sequence
            dropped_joints = np.random.choice(
                [True, False], 
                size=(num_joints,), 
                p=[self.joint_dropout, 1.0 - self.joint_dropout]
            )
            if dropped_joints.any():
                coords[:, :, dropped_joints, :] = 0
                scores[:, :, dropped_joints, :] = 0
                
        # Re-assemble x
        x = np.concatenate([coords, scores], axis=0)
        return x
