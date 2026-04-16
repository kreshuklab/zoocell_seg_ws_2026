# Standard scientific computing libraries
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Image I/O and processing
import tifffile
import h5py

# Deep learning framework
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# BioImage.IO model loading
import bioimageio.core
from bioimageio.core import load_description

# Data Loading Utilities

def load_volumetric_data(file_path, time_idx=0, channel_idx=0, out_axes='ZYX', internal_path=None, 
                         series_idx=0, normalize=True, invert_contrast=False, load_labels=False):
    """
    Robust loader for volumetric microscopy data.

    Parameters:
    - file_path: Path to the data file
    - time_idx: Time point index (for time-lapse data) - not applicable for static volumes (us)
    - channel_idx: Channel index (for multi-channel data)
    - out_axes: Desired output axis order ('ZYX' for z,y,x)
    - internal_path: Internal path within the file (for formats that support it)

    Returns:
    - volume: Normalized 3D array
    - metadata: Dictionary with file info
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        if file_ext in ['.tif', '.tiff']:
            # Use tifffile for TIFF files
            with tifffile.TiffFile(file_path, is_ome=False) as tif:
                if tif.is_ome:
                    # OME-TIFF handling
                    data = tif.asarray(series=series_idx)
                    # Handle different axis orders
                    if data.ndim == 4:  # TZYX or TCYX
                        data = data[time_idx, channel_idx] if 'T' in tif.series[series_idx].axes else data[channel_idx, time_idx]
                    elif data.ndim == 3:  # ZYX or CYX
                        data = data[channel_idx] if 'C' in tif.series[series_idx].axes else data
                else:
                    # Regular TIFF stack
                    data = tifffile.imread(file_path, level=series_idx, is_ome=False)
                    #data = tiff_file.asarray()

        elif file_ext in ['.h5', '.hdf5']:
            # HDF5 handling
            with h5py.File(file_path, 'r') as f:
                if internal_path is not None:
                    if internal_path not in f:
                        raise KeyError(f"internal_path '{internal_path}' not found in {file_path}")
                    if load_labels:
                        data = f[internal_path][()].astype(np.float32)  # Load labels as float32 for processing
                    else:
                        data = f[internal_path][:]
                elif 'data' in f:
                    data = f['data'][:]
                else:
                    key = list(f.keys())[0]
                    data = f[key][:]

        elif file_ext == '.n5':
            # N5 format (similar to HDF5)
            import zarr
            store = zarr.N5Store(file_path)
            root = zarr.open(store, mode='r')
            if internal_path is not None:
                if internal_path not in root:
                    raise KeyError(f"internal_path '{internal_path}' not found in {file_path}")
                if load_labels:
                    data = root[internal_path][()].astype(np.float32)  # Load labels as float32 for processing
                else:
                    data = root[internal_path][:]
            elif 'data' in root:
                data = root['data'][:]
            else:
                key = list(root.keys())[0]
                data = root[key][:]

        else:
            # Try bioio as fallback
            img = BioImage(file_path)
            data = img.data
            if internal_path is not None:
                # If path indexing is possible for this object, use it; else ignore
                try:
                    data = img.data[internal_path]
                except Exception:
                    pass
            if data.ndim > 3:
                data = data[time_idx, channel_idx]

    except Exception as e:
        raise RuntimeError(f"Failed to load {file_path}: {str(e)}")

    # Ensure we have a 3D volume
    if data.ndim != 3:
        raise ValueError(f"Expected 3D data, got {data.ndim}D array with shape {data.shape}")

    # Normalize data type and range
    if load_labels == True and data.dtype != np.float32:
        print(f"Converting data from {data.dtype} to float32 for processing...")
        data = data.astype(np.float32)

    # Normalize to [0, 1] range
    data_min, data_max = np.percentile(data, [1, 99])  # Robust normalization
    if normalize:
        if load_labels == True:
            data = (data - data_min) / (data_max - data_min + 1e-8)
        else:    
            if data_max > data_min:
                if invert_contrast:
                    data = ((data - data_min) / (data_max - data_min)) * -1 + 1  # Invert contrast for EM data (optional)
                else:
                    data = (data - data_min) / (data_max - data_min)
            data = np.clip(data, 0, 1)

    # Reorder axes if necessary
    current_axes = 'ZYX'  # Assume input is ZYX
    if out_axes != current_axes:
        # Simple axis reordering (extend for more complex cases)
        print(f"Reordering axes from {current_axes} to {out_axes}...")
        axis_map = {'Z': 0, 'Y': 1, 'X': 2}
        target_order = [axis_map[ax] for ax in out_axes]
        data = np.transpose(data, target_order)

    if normalize:
        metadata = {
            'original_shape': data.shape,
            'dtype': str(data.dtype),
            'file_path': file_path,
            'axes': out_axes,
            'normalization_range': (data_min, data_max)
        }
    else:
        metadata = {
            'original_shape': data.shape,
            'dtype': str(data.dtype),
            'file_path': file_path,
            'axes': out_axes
        }

    return data, metadata

def visualize_volume_slices(volume, title="Volume Slices", num_slices=5, cmap='gray', interpolation='none'):
    """
    Visualize multiple slices through a 3D volume.
    """
    fig, axes = plt.subplots(1, num_slices, figsize=(4*num_slices, 4))

    if num_slices == 1:
        axes = [axes]

    slice_indices = np.linspace(0, volume.shape[0]-1, num_slices, dtype=int)

    for i, z_idx in enumerate(slice_indices):
        axes[i].imshow(volume[z_idx], cmap=cmap, interpolation=interpolation)
        axes[i].set_title(f"Z = {z_idx}")
        axes[i].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Create a colored version of labels where each instance gets a random color
def create_colored_labels(label_volume):
    """Assign random colors to each label ID for visualization."""
    unique_labels_list = np.unique(label_volume)
    num_labels = len(unique_labels_list)
    
    # Create random color map (skip background 0)
    colors = np.random.rand(num_labels, 3)
    colors[0] = [0, 0, 0]  # Background is black
    
    colored_volume = np.zeros((*label_volume.shape, 3), dtype=np.float32)
    for idx, label_id in enumerate(unique_labels_list):
        mask = label_volume == label_id
        colored_volume[mask] = colors[idx]
    
    return colored_volume, colors, unique_labels_list

from skimage.morphology import binary_closing, binary_opening, ball
from scipy.ndimage import gaussian_filter
# Smoothing helper for binary boundaries
def smooth_binary_boundaries(boundaries, gaussian_sigma=1.0, morph_radius=2, threshold=0.15):
    """Smooth binary boundaries in 3D using Gaussian blur followed by morphological operations."""
    # Gaussian smoothing softens the edges
    blurred = gaussian_filter(boundaries.astype(np.float32), sigma=gaussian_sigma)

    # Threshold back to binary to preserve boundary mask
    binary = (blurred > threshold)

    # Morphological smooth: closing then opening reduces jaggedness and holes
    selem = ball(morph_radius)
    smoothed = binary_closing(binary, footprint=selem)
    smoothed = binary_opening(smoothed, footprint=selem)

    return smoothed.astype(np.float32)

def visualize_boundary_conversion(raw_vol, label_vol, boundary_vol):
    z_idx = random.randint(0, raw_vol.shape[0] - 1)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. Raw Data
    axes[0].imshow(raw_vol[z_idx], cmap='gray')
    axes[0].set_title(f"Raw Data (Z={z_idx})")
    axes[0].axis('off')
    
    # 2. Original Color Annotations (Instance Mask)
    # Using 'nipy_spectral' to give distinct IDs starkly different colors
    axes[1].imshow(label_vol[z_idx], cmap='viridis', interpolation='nearest')
    axes[1].set_title("Your Manual Annotations (Labels)")
    axes[1].axis('off')
    
    # 3. Extracted Binary Boundaries (The actual Ground Truth)
    axes[2].imshow(boundary_vol[z_idx], cmap='gray')
    axes[2].set_title("Extracted Binary Boundaries (Model GT)")
    axes[2].axis('off')
    
    # 4. Overlay of Raw and Boundaries
    axes[3].imshow(raw_vol[z_idx], cmap='gray')
    axes[3].imshow(boundary_vol[z_idx], cmap='magma', alpha=0.4)
    axes[3].set_title("Raw + Boundary Overlay")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
import sys
def load_cebraem_model(model_weights_path="/g/kreshuk/frazer/tools/CebraEM/models/cebraEM_repackaged/model_0097.pt",
                       rdf_path="/g/kreshuk/frazer/tools/CebraEM/models/cebraEM_repackaged/rdf.yaml",
                       model_architecture_path="/g/kreshuk/frazer/tools/CebraEM/models/cebraEM_repackaged",
                       device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load CebraEM pre-trained model with its weights.

    Parameters:
    - model_weights_path: Path to PyTorch weights (.pt file)
    - rdf_path: Path to RDF metadata file
    - device: Computing device ('cpu' or 'cuda')

    Returns:
    - model: Loaded PyTorch PiledUnet model with weights
    - metadata: Model information from RDF
    """
    
    sys.path.insert(0, model_architecture_path)  # Ensure the model architecture can be imported
    
    try:
        from piled_unets import PiledUnet
    except ImportError as e:
        raise ImportError(f"Failed to import PiledUnet from {model_architecture_path}: {str(e)}")

    try:
        print(f"Loading CebraEM model from: {model_weights_path}")
        print(f"Loading metadata from: {rdf_path}")
        try:
            # Load metadata from RDF
            model_description = load_description(rdf_path)

            print("✅ Model loaded successfully!")

            # Extract metadata
            metadata = {
                'name': model_description.name,
                'description': model_description.description,
                'authors': [author.name for author in model_description.authors],
                'license': model_description.license,
                'input_shape': model_description.inputs[0].shape,
                'output_shape': model_description.outputs[0].shape,
                'input_axes': model_description.inputs[0].axes,
                'output_axes': model_description.outputs[0].axes,
            }

            print(f'The model description is: {model_description}')
        except:
            # If load_bioimage_model_for_finetuning fails, extract minimal metadata
            metadata = {'name': 'CebraEM', 'fallback': False}
        
        # Instantiate the PiledUnet model with the architecture specified in RDF
        model = PiledUnet(
            n_nets=3,
            in_channels=1,
            out_channels=[1, 1, 1],  # 3 U-Nets, each outputs 1 channel
            filter_sizes_down=(
                ((8, 16), (16, 32), (32, 64)),
                ((8, 16), (16, 32), (32, 64)),
                ((8, 16), (16, 32), (32, 64))
            ),
            filter_sizes_bottleneck=(
                (64, 128),
                (64, 128),
                (64, 128)
            ),
            filter_sizes_up=(
                ((64, 64), (32, 32), (16, 16)),
                ((64, 64), (32, 32), (16, 16)),
                ((64, 64), (32, 32), (16, 16))
            ),
            batch_norm=True,
            output_activation='sigmoid',
            predict=True
        )
        
        # Load weights from .pt file
        print(f"Loading weights from: {model_weights_path}")
        state_dict = torch.load(model_weights_path, map_location=device)
        model.load_state_dict(state_dict)
        
        # Move model to device
        model = model.to(device)
        model.eval()
        
        print(f"✅ CebraEM model loaded successfully on device: {device}")
        
        metadata.update({
            'model_type': 'PiledUnet',
            'n_nets': 3,
            'batch_norm': True,
            'weights_path': model_weights_path,
            'input_shape': (1, 1, 128, 128, 128),  # Min size from RDF
            'output_shape': (1, 1, 64, 64, 64),     # With halo=32
        })
        
        return model, metadata, device

    except Exception as e:        
        raise RuntimeError(f"Failed to load CebraEM model:\n{str(e)}")
    
    
# Sliding Window Inference for Large Volumes

def sliding_window_inference(model, volume, patch_size=(128, 128, 128), overlap=(32, 32, 32),
                           batch_size=1, device='cpu', verbose=True):
    """
    Perform sliding window inference on large 3D volumes.

    Parameters:
    - model: PyTorch model for prediction
    - volume: 3D input volume (ZYX)
    - patch_size: Size of patches to process
    - overlap: Overlap between patches
    - batch_size: Patches per batch
    - device: Computing device
    - verbose: Print progress

    Returns:
    - prediction: 3D prediction volume
    """
    if verbose:
        print(f"Starting sliding window inference...")
        print(f"Volume shape: {volume.shape}")
        print(f"Patch size: {patch_size}")
        print(f"Overlap: {overlap}")

    # Ensure volume is float32 and add channel dimension
    if volume.dtype != np.float32:
        volume = volume.astype(np.float32)
    if volume.ndim == 3:
        volume = volume[np.newaxis]  # Add channel dim: CZYX

    C, Z, Y, X = volume.shape
    pz, py, px = patch_size
    oz, oy, ox = overlap

    # Calculate number of patches
    nz = int(np.ceil((Z - pz) / (pz - oz))) + 1 if Z > pz else 1
    ny = int(np.ceil((Y - py) / (py - oy))) + 1 if Y > py else 1
    nx = int(np.ceil((X - px) / (px - ox))) + 1 if X > px else 1

    if verbose:
        print(f"Number of patches: {nz} x {ny} x {nx} = {nz*ny*nx}")

    # Initialize prediction volume
    prediction = np.zeros((1, Z, Y, X), dtype=np.float32)
    weight_map = np.zeros((1, Z, Y, X), dtype=np.float32)

    # Generate patch coordinates
    patch_coords = []
    for iz in range(nz):
        z_start = min(iz * (pz - oz), Z - pz)
        z_end = z_start + pz
        for iy in range(ny):
            y_start = min(iy * (py - oy), Y - py)
            y_end = y_start + py
            for ix in range(nx):
                x_start = min(ix * (px - ox), X - px)
                x_end = x_start + px
                patch_coords.append((z_start, z_end, y_start, y_end, x_start, x_end))

    # Process patches in batches
    model.eval()
    with torch.no_grad():
        for i in range(0, len(patch_coords), batch_size):
            batch_coords = patch_coords[i:i+batch_size]
            batch_patches = []

            for z_start, z_end, y_start, y_end, x_start, x_end in batch_coords:
                patch = volume[:, z_start:z_end, y_start:y_end, x_start:x_end]
                batch_patches.append(torch.from_numpy(patch).float())

            # Stack and move to device
            batch_tensor = torch.stack(batch_patches).to(device)

            # Forward pass
            batch_pred = model(batch_tensor)  # PiledUnet already applies sigmoid

            # Move back to CPU
            batch_pred = batch_pred.cpu().numpy()

            # Accumulate predictions
            for j, (z_start, z_end, y_start, y_end, x_start, x_end) in enumerate(batch_coords):
                pred_patch = batch_pred[j]
                prediction[:, z_start:z_end, y_start:y_end, x_start:x_end] += pred_patch
                weight_map[:, z_start:z_end, y_start:y_end, x_start:x_end] += 1

            if verbose and (i // batch_size) % 10 == 0:
                progress = (i + len(batch_coords)) / len(patch_coords) * 100
                print(f"Progress: {progress:.1f}%")

    # Normalize by weight map
    prediction = np.divide(prediction, weight_map, out=np.zeros_like(prediction), where=weight_map!=0)

    # Remove channel dimension
    prediction = prediction[0]  # ZYX

    if verbose:
        print("✅ Sliding window inference complete!")

    return prediction

# Ground-Truth Data Preparation

class PatchDataset(Dataset):
    """
    Dataset for loading 3D patches from volumetric data with corresponding labels.
    """
    def __init__(self, volume, labels, patch_size=(128, 128, 128), samples_per_epoch=100,
                 augment=True, transform=None):
        self.volume = volume
        self.labels = labels
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.transform = transform

        # Ensure same shape
        assert volume.shape == labels.shape, f"Volume {volume.shape} and labels {labels.shape} must have same shape"

        self.Z, self.Y, self.X = volume.shape

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # Random patch extraction
        pz, py, px = self.patch_size

        # Random position (ensure patch fits)
        z_start = np.random.randint(0, max(1, self.Z - pz))
        y_start = np.random.randint(0, max(1, self.Y - py))
        x_start = np.random.randint(0, max(1, self.X - px))

        z_end = min(z_start + pz, self.Z)
        y_end = min(y_start + py, self.Y)
        x_end = min(x_start + px, self.X)

        # Extract patch
        vol_patch = self.volume[z_start:z_end, y_start:y_end, x_start:x_end]
        label_patch = self.labels[z_start:z_end, y_start:y_end, x_start:x_end]

        # Pad if necessary
        if vol_patch.shape != self.patch_size:
            vol_patch = np.pad(vol_patch, [(0, pz - vol_patch.shape[0]),
                                          (0, py - vol_patch.shape[1]),
                                          (0, px - vol_patch.shape[2])], mode='reflect')
            label_patch = np.pad(label_patch, [(0, pz - label_patch.shape[0]),
                                              (0, py - label_patch.shape[1]),
                                              (0, px - label_patch.shape[2])], mode='reflect')

        # Add channel dimension
        vol_patch = vol_patch[np.newaxis]  # CZYX
        label_patch = label_patch[np.newaxis]

        # Convert to tensors
        vol_tensor = torch.from_numpy(vol_patch).float()
        label_tensor = torch.from_numpy(label_patch).float()

        # Apply transforms if provided
        if self.transform:
            vol_tensor, label_tensor = self.transform(vol_tensor, label_tensor)

        return vol_tensor, label_tensor


# 3D Data Augmentation using TorchVision Transforms

class Compose3D:
    """Composes several 3D transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, volume, label):
        for t in self.transforms:
            volume, label = t(volume, label)
        return volume, label


class RandomHorizontalFlip3D:
    """Random horizontal flip for 3D data."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, volume, label):
        if torch.rand(1) < self.p:
            # Flip along Y axis (horizontal)
            volume = torch.flip(volume, dims=[2])
            label = torch.flip(label, dims=[2])
        return volume, label


class RandomVerticalFlip3D:
    """Random vertical flip for 3D data."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, volume, label):
        if torch.rand(1) < self.p:
            # Flip along X axis (vertical)
            volume = torch.flip(volume, dims=[3])
            label = torch.flip(label, dims=[3])
        return volume, label


class RandomRotation3D:
    """Random rotation for 3D data (90, 180, 270 degrees)."""
    def __init__(self, degrees=[0, 90, 180, 270], p=0.5):
        self.degrees = degrees
        self.p = p

    def __call__(self, volume, label):
        if torch.rand(1) < self.p:
            # Choose random rotation
            angle = torch.tensor(np.random.choice(self.degrees))
            k = int(angle / 90)  # Convert to number of 90-degree rotations

            if k > 0:
                # Rotate in ZY plane (keeping X fixed)
                volume = torch.rot90(volume, k=k, dims=[1, 2])
                label = torch.rot90(label, k=k, dims=[1, 2])
        return volume, label


class RandomAffine3D:
    """Random affine transformation for 3D data."""
    def __init__(self, degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), p=0.5):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.p = p

    def __call__(self, volume, label):
        if torch.rand(1) < self.p:
            # For 3D, we'll apply 2D affine transforms slice by slice
            C, D, H, W = volume.shape

            # Random parameters
            angle = torch.FloatTensor(1).uniform_(-self.degrees, self.degrees).item()
            tx = torch.FloatTensor(1).uniform_(-self.translate[0], self.translate[0]).item() * W
            ty = torch.FloatTensor(1).uniform_(-self.translate[1], self.translate[1]).item() * H
            sx = torch.FloatTensor(1).uniform_(self.scale[0], self.scale[1]).item()
            sy = torch.FloatTensor(1).uniform_(self.scale[0], self.scale[1]).item()

            # Apply to each slice along Z dimension
            transformed_volume = torch.zeros_like(volume)
            transformed_label = torch.zeros_like(label)

            for z in range(D):
                # Volume slice
                vol_slice = volume[0, z, :, :]  # Remove channel dim for torchvision
                vol_slice = transforms.functional.affine(
                    vol_slice.unsqueeze(0), angle=angle,
                    translate=[tx, ty], scale=sx, shear=0
                )
                transformed_volume[0, z, :, :] = vol_slice[0]

                # Label slice
                label_slice = label[0, z, :, :]
                label_slice = transforms.functional.affine(
                    label_slice.unsqueeze(0), angle=angle,
                    translate=[tx, ty], scale=sy, shear=0
                )
                transformed_label[0, z, :, :] = label_slice[0]

            volume, label = transformed_volume, transformed_label
        return volume, label


class ColorJitter3D:
    """Random color jitter for 3D volume data."""
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5):
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness, contrast=contrast,
            saturation=saturation, hue=hue
        )
        self.p = p

    def __call__(self, volume, label):
        if torch.rand(1) < self.p:
            C, D, H, W = volume.shape
            # Apply color jitter slice by slice
            for z in range(D):
                vol_slice = volume[0, z, :, :].unsqueeze(0)  # Add channel dim
                vol_slice = self.color_jitter(vol_slice)
                volume[0, z, :, :] = vol_slice[0]
        return volume, label


class GaussianNoise3D:
    """Add Gaussian noise to 3D volume."""
    def __init__(self, mean=0, std=0.05, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, volume, label):
        if torch.rand(1) < self.p:
            noise = torch.randn_like(volume) * self.std + self.mean
            volume = torch.clamp(volume + noise, 0, 1)
        return volume, label


class ElasticDeformation3D:
    """Elastic deformation for 3D data (simplified version)."""
    def __init__(self, alpha=50, sigma=5, p=0.3):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, volume, label):
        if torch.rand(1) < self.p:
            C, D, H, W = volume.shape

            # Generate displacement fields
            dx = torch.randn(D, H, W) * self.alpha
            dy = torch.randn(D, H, W) * self.alpha

            # Smooth displacements
            dx = ndimage.gaussian_filter(dx.numpy(), sigma=self.sigma)
            dy = ndimage.gaussian_filter(dy.numpy(), sigma=self.sigma)

            dx = torch.from_numpy(dx).float()
            dy = torch.from_numpy(dy).float()

            # Apply deformation slice by slice
            for z in range(D):
                # Volume deformation
                vol_slice = volume[0, z, :, :]
                vol_deformed = ndimage.map_coordinates(
                    vol_slice.numpy(), [np.arange(H) + dy[z], np.arange(W) + dx[z]],
                    order=1, mode='reflect'
                )
                volume[0, z, :, :] = torch.from_numpy(vol_deformed).float()

                # Label deformation (nearest neighbor for labels)
                label_slice = label[0, z, :, :]
                label_deformed = ndimage.map_coordinates(
                    label_slice.numpy(), [np.arange(H) + dy[z], np.arange(W) + dx[z]],
                    order=0, mode='reflect'
                )
                label[0, z, :, :] = torch.from_numpy(label_deformed).float()

        return volume, label


# Create robust augmentation pipeline
def get_training_transforms():
    """Get the standard training augmentation transforms."""
    return Compose3D([
        RandomHorizontalFlip3D(p=0.5),
        RandomVerticalFlip3D(p=0.5),
        RandomRotation3D(p=0.5),
        ColorJitter3D(brightness=0.1, contrast=0.1, p=0.3),
        GaussianNoise3D(std=0.02, p=0.3),
        #lasticDeformation3D(alpha=30, sigma=4, p=0.2)
    ])

# Fine-Tuning Implementation

def dice_loss(pred_logits, targets, smooth=1e-5):
    """
    Dice loss for boundary segmentation.
    """
    pred_probs = torch.sigmoid(pred_logits)
    intersection = (pred_probs * targets).sum(dim=(1, 2, 3, 4))
    union = pred_probs.sum(dim=(1, 2, 3, 4)) + targets.sum(dim=(1, 2, 3, 4))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def combined_loss(pred_logits, targets, bce_weight=1.0, dice_weight=1.0):
    """
    Combined BCE + Dice loss.
    """
    bce = nn.BCEWithLogitsLoss()(pred_logits, targets)
    dice = dice_loss(pred_logits, targets)
    return bce_weight * bce + dice_weight * dice

def fine_tune_model(model, train_loader, num_epochs=10, learning_rate=1e-4,
                   device='cpu', freeze_encoder=True):
    """
    Fine-tune the model on custom data.

    Parameters:
    - model: PyTorch model to fine-tune
    - train_loader: DataLoader with training data
    - num_epochs: Number of training epochs
    - learning_rate: Initial learning rate
    - device: Computing device
    - freeze_encoder: Whether to freeze early layers

    Returns:
    - model: Fine-tuned model
    - training_history: Loss history
    """
    # Freeze encoder layers if requested
    if freeze_encoder and hasattr(model, 'encoder') or hasattr(model, 'enc1'):
        print("Freezing encoder layers...")
        for name, param in model.named_parameters():
            if 'enc' in name or 'encoder' in name:
                param.requires_grad = False

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params}/{total_params} ({100*trainable_params/total_params:.1f}%)")

    # Setup optimizer and scheduler
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training loop
    model.train()
    training_history = []

    print("Starting fine-tuning...")
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_vol, batch_labels in train_loader:
            batch_vol = batch_vol.to(device)
            batch_labels = batch_labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            predictions = model(batch_vol)
            loss = combined_loss(predictions, batch_labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        # Update learning rate
        scheduler.step()

        # Record metrics
        avg_loss = epoch_loss / num_batches
        current_lr = optimizer.param_groups[0]['lr']
        training_history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'learning_rate': current_lr
        })

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

    print("✅ Fine-tuning complete!")
    
        # Plot training history
    losses = [h['loss'] for h in training_history]
    lrs = [h['learning_rate'] for h in training_history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(losses, 'b-o')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)

    ax2.plot(lrs, 'r-o')
    ax2.set_title('Learning Rate Schedule')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_yscale('log')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
    
    return model, training_history


def visualize_model_comparison(volume, original_prediction, finetuned_prediction, slice_idx=None):
    # Visualize comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    if slice_idx is None:
        slice_idx = volume.shape[0] // 2

    # Row 1: Original model
    axes[0,0].imshow(volume[slice_idx], cmap='gray')
    axes[0,0].set_title("Raw EM Image")

    axes[0,1].imshow(original_prediction[slice_idx], cmap='viridis', vmin=0, vmax=1)
    axes[0,1].set_title("Original Model Prediction")

    axes[0,2].imshow(volume[slice_idx], cmap='gray')
    axes[0,2].imshow(original_prediction[slice_idx], cmap='viridis', alpha=0.5, vmin=0, vmax=1)
    axes[0,2].set_title("Original Model Overlay")

    # Row 2: Fine-tuned model
    axes[1,0].imshow(volume[slice_idx], cmap='gray')
    axes[1,0].set_title("Raw EM Image")

    axes[1,1].imshow(finetuned_prediction[slice_idx], cmap='viridis', vmin=0, vmax=1)
    axes[1,1].set_title("Fine-tuned Model Prediction")

    axes[1,2].imshow(volume[slice_idx], cmap='gray')
    axes[1,2].imshow(finetuned_prediction[slice_idx], cmap='viridis', alpha=0.5, vmin=0, vmax=1)
    axes[1,2].set_title("Fine-tuned Model Overlay")

    plt.tight_layout()
    plt.show()
    

# Instance Segmentation
def apply_watershed_segmentation(boundary_prediction, threshold=0.5, sigma_seeds=2.0, min_size=50):
    """
    Apply watershed segmentation to boundary predictions.

    Parameters:
    - boundary_prediction: Boundary probability map [0,1]
    - threshold: Boundary threshold
    - sigma_seeds: Seed smoothing sigma
    - min_size: Minimum segment size

    Returns:
    - segmentation: Labeled segmentation
    """
    # Segmentation algorithms
    import elf.segmentation.watershed as ws
    import elf.segmentation.features as feats
    import elf.segmentation.multicut as mc
    print("Applying watershed segmentation...")

    watershed_result = ws.distance_transform_watershed(
        boundary_prediction,
        threshold=threshold,
        sigma_seeds=sigma_seeds,
        min_size=min_size
    )

    # Handle tuple return
    if isinstance(watershed_result, tuple):
        segmentation = watershed_result[0]
    else:
        segmentation = watershed_result

    num_cells = len(np.unique(segmentation)) - 1  # Subtract background
    print(f"Found {num_cells} cells with watershed")

    return segmentation

def apply_multicut_segmentation(boundary_prediction, beta=0.5, threshold=0.5, sigma_seeds=2.0, min_size=50):
    """
    Apply multicut segmentation to boundary predictions.

    Parameters:
    - boundary_prediction: Boundary probability map [0,1]
    - beta: Multicut beta parameter
    - threshold: Initial oversegmentation threshold

    Returns:
    - segmentation: Labeled segmentation
    """
    # Segmentation algorithms
    import elf.segmentation.watershed as ws
    import elf.segmentation.features as feats
    import elf.segmentation.multicut as mc
    print("Applying multicut segmentation...")

    # Create supervoxels (oversegmentation)
    super_result = ws.distance_transform_watershed(
        boundary_prediction,
        threshold=threshold,
        sigma_seeds=sigma_seeds,
        min_size=min_size
    )

    if isinstance(super_result, tuple):
        supervoxels = super_result[0]
    else:
        supervoxels = super_result

    # Build region adjacency graph
    rag = feats.compute_rag(supervoxels)
    edge_features = feats.compute_boundary_features(rag, boundary_prediction)
    edge_probabilities = edge_features[:, 0]

    # Compute edge costs
    edge_costs = mc.compute_edge_costs(edge_probabilities, beta=beta)

    # Solve multicut
    node_labels = mc.multicut_kernighan_lin(rag, edge_costs)

    # Project back to pixels
    segmentation = feats.project_node_labels_to_pixels(rag, node_labels)

    num_cells = len(np.unique(segmentation)) - 1
    print(f"Found {num_cells} cells with multicut (beta={beta})")

    return segmentation


def visualize_segmentation_comparison(volume, prediction, watershed_seg, multicut_seg, slice_idx=None):
    cmap = ListedColormap(np.random.rand(256, 3))
    cmap.colors[0] = [0, 0, 0]  # Black background

    fig, axes = plt.subplots(1, 4, figsize=(18, 12))

    if slice_idx is None:
        slice_idx = volume.shape[0] // 2

    # Row 1: Watershed + Multicut
    axes[0].imshow(volume[slice_idx], cmap='gray')
    axes[0].set_title("Raw EM Image")

    axes[1].imshow(prediction[slice_idx], cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title("Boundary Prediction")

    axes[2].imshow(watershed_seg[slice_idx], cmap=cmap, interpolation='nearest')
    axes[2].set_title(f"Watershed Segmentation\n({len(np.unique(watershed_seg))-1} cells)")

    axes[3].imshow(multicut_seg[slice_idx], cmap=cmap, interpolation='nearest')
    axes[3].set_title(f"Multicut Segmentation\n({len(np.unique(multicut_seg))-1} cells)")

    plt.tight_layout()
    plt.show()

# Saving and Exporting Results

def save_segmentation_results(segmentation, watershed_seg, boundary_pred, output_path, metadata=None):
    """
    Save segmentation results to disk.

    Parameters:
    - segmentation: Instance segmentation (labeled image)
    - watershed_seg: Watershed segmentation
    - boundary_pred: Boundary predictions
    - output_path: Output file path
    - metadata: Additional metadata to save
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('segmentation', data=segmentation, compression='gzip')
        f.create_dataset('watershed_segmentation', data=watershed_seg, compression='gzip')
        f.create_dataset('boundaries', data=boundary_pred, compression='gzip')

        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float)):
                    f.attrs[key] = value
                elif isinstance(value, (list, tuple)) and all(isinstance(x, (str, int, float)) for x in value):
                    f.attrs[key] = str(value)

    print(f"Results saved to: {output_path}")
    
    
    from pathlib import Path

# BioImage.IO imports
from bioimageio.spec.model.v0_5 import (
    ModelDescr,
    WeightsDescr,
    FileDescr, 
    PytorchStateDictWeightsDescr,
    TorchscriptWeightsDescr,
    ArchitectureFromFileDescr,
    InputTensorDescr,
    OutputTensorDescr,
    Author,
    CiteEntry
)
from bioimageio.spec import save_bioimageio_package
from bioimageio.core import test_model

def save_model_for_bioimageio(
    model: torch.nn.Module, 
    model_name: str = "test_package_CebraNet",
    output_dir: str = "./exported_ft_models", 
    author_name: str = "Seth Frazer",
    author_email: str = "seth.frazer@embl.de",
    input_shape: tuple = (1, 1, 128, 128, 128),
    model_version: str = "1.1.0",
    architecture_file: str = "tools/CebraEM/models/cebraEM_finetuned/piled_unets.py",
    architecture_callable: str = "PiledUnet",
    architecture_kwargs: dict = None
):
    """
    Export a PyTorch model into a fully validated BioImage.IO v0.5 package (.zip),
    including correct tracing and preprocessing specifications.
    """
    
    from pathlib import Path
    # Default architecture kwargs to match your CebraNet specifications
    if architecture_kwargs is None:
        architecture_kwargs = {
            "n_nets": 3,
            "in_channels": 1,
            "out_channels": [1, 1, 1],
            "filter_sizes_down": [
                [[8, 16], [16, 32], [32, 64]],
                [[8, 16], [16, 32], [32, 64]],
                [[8, 16], [16, 32], [32, 64]]
            ],
            "filter_sizes_bottleneck": [
                [64, 128],
                [64, 128],
                [64, 128]
            ],
            "filter_sizes_up": [
                [[64, 64], [32, 32], [16, 16]],
                [[64, 64], [32, 32], [16, 16]],
                [[64, 64], [32, 32], [16, 16]]
            ],
            "batch_norm": True,
            "output_activation": "sigmoid"
        }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Starting BioImage.IO Model Export...")
    print("=" * 70)

    # ==================== 1. GENERATE TEST TENSORS ====================
    print("\n1️⃣  Generating required test tensors (uint8 -> float32)...")
    
    test_input_path = out_dir / "test_input.npy"
    test_output_path = out_dir / "test_output.npy"
    
    # Safely move model to CPU and set to eval mode for tracing and testing
    model.cpu()
    model.eval()
    
    # Create uint8 test input as specified in your script/YAML
    test_input_uint8 = np.random.randint(0, 255, size=input_shape, dtype=np.uint8)
    np.save(test_input_path, test_input_uint8)
    
    # Simulate scale_linear preprocessing natively for the PyTorch forward pass
    gain = 0.00392156862745098
    test_input_scaled = test_input_uint8.astype(np.float32) * gain
    test_input_tensor = torch.from_numpy(test_input_scaled)
    
    with torch.no_grad():
        test_output = model(test_input_tensor)
        
    np.save(test_output_path, test_output.numpy())
    print(f"   ✅ Test arrays saved: {test_input_path.name}, {test_output_path.name}")
    
    # BioImage.IO strictly requires a markdown documentation file
    doc_path = out_dir / "CebraNET_README.md"
    with open(doc_path, "w") as f:
        f.write(f"# {model_name}\n\nFine-tuned EM cell segmentation model.")
    print(f"   ✅ Documentation stub saved: {doc_path.name}")
    
    # ==================== 2. SAVE MODEL WEIGHTS & TRACING ====================
    print("\n2️⃣  Saving PyTorch and Tracing TorchScript weights...")
    
    # A. Save standard PyTorch state_dict
    pytorch_path = out_dir / f"{model_name}.pth"
    torch.save(model.state_dict(), pytorch_path)
    print(f"   ✅ PyTorch weights saved: {pytorch_path.name}")
    
    # B. Trace and save TorchScript using the generated tensor
    torchscript_path = out_dir / f"{model_name}_torchscript.pt"
    try:
        traced_model = torch.jit.trace(model, test_input_tensor)
        traced_model.save(torchscript_path)
        print(f"   ✅ TorchScript weights saved (traced): {torchscript_path.name}")
    except Exception as e:
        print(f"   ⚠️  TorchScript export failed: {e}. Model package will only contain PyTorch weights.")
        torchscript_path = None

    # ==================== 3. CREATE BIOIMAGE.IO DESCRIPTORS ====================
    print("\n3️⃣  Building BioImage.IO Descriptor Classes (v0.5)...")
    
    # Define the weights container (Including both PyTorch and TorchScript)
    weights_kwargs = {
        "pytorch_state_dict": PytorchStateDictWeightsDescr(
            source=pytorch_path,
            architecture=ArchitectureFromFileDescr(
                source=architecture_file,
                callable=architecture_callable,
                kwargs=architecture_kwargs
            ),
            pytorch_version=torch.__version__
        )
    }
    
    if torchscript_path:
        weights_kwargs["torchscript"] = TorchscriptWeightsDescr(
            source=torchscript_path,
            pytorch_version=torch.__version__
        )
        
    weights_descr = WeightsDescr(**weights_kwargs)

    # Define Input Tensor Structure matching rdf.yaml
    input_tensor = InputTensorDescr(
        id="input0",
        test_tensor=FileDescr(source=test_input_path),
        #data_type="uint8",
        preprocessing=[{"id": "scale_linear", "kwargs": {"gain": gain}}],
        axes=[
            {"type": "batch", "id": "b"},
            {"type": "channel", "id": "c", "channel_names": ["channel0"]},
            {"type": "space", "id": "z", "size": {"min": 128, "step": 16}},
            {"type": "space", "id": "y", "size": {"min": 128, "step": 16}},
            {"type": "space", "id": "x", "size": {"min": 128, "step": 16}}
        ],
    )
    
    # Define Output Tensor Structure matching rdf.yaml
    output_tensor = OutputTensorDescr(
        id="output0",
        test_tensor=FileDescr(source=test_output_path),
        #data_type="float32",
        axes=[
            {"type": "batch", "id": "b"},
            {"type": "channel", "id": "c", "channel_names": ["channel0"]},
            {"type": "space", "id": "z", "size": {"tensor_id": "input0", "axis_id": "z"}, "halo": 32},
            {"type": "space", "id": "y", "size": {"tensor_id": "input0", "axis_id": "y"}, "halo": 32},
            {"type": "space", "id": "x", "size": {"tensor_id": "input0", "axis_id": "x"}, "halo": 32}
        ]
    )

    # Assemble the final model description
    model_description = ModelDescr(
        id="joyful-deer",
        id_emoji="🦌",
        name=model_name,
        version=model_version,
        description="Fine-tuned volumetric EM segmentation model.",
        authors=[Author(name=author_name, email=author_email)],
        cite=[CiteEntry(
            text="Hennies et al. 2023, CebraEM: A practical workflow to segment cellular organelles in volume SEM datasets using a transferable CNN-based membrane prediction", 
            doi="10.1101/2023.04.06.535829"
        )],
        tags=["unet", "3d", "cells", "whole-organism", "ilastik", "semantic-segmentation", 
              "electron-microscopy", "pytorch", "membranes", "hela", "macrophage", "platynereis", "calu-3"],
        license="CC-BY-4.0",
        documentation=doc_path,
        inputs=[input_tensor],
        outputs=[output_tensor],
        weights=weights_descr,
    )

    # ==================== 4. PACKAGE AND VALIDATE ====================
    print("\n4️⃣  Packaging into BioImage.IO format (.zip) & Validating...")
    
    zip_path = out_dir / f"{model_name}.zip"
    
    # Computes SHA256 hashes, writes the yaml, and creates the zip automatically
    package_path = save_bioimageio_package(model_description, output_path=zip_path)
    print(f"   ✅ Model packaged successfully to: {package_path}")
    
    # Run bioimageio validation
    try:
        validation_summary = test_model(package_path)
        
        report_path = out_dir / "validation_report.txt"
        with open(report_path, "w") as f:
            f.write(str(validation_summary))
            
        if validation_summary.status == 'passed':
            print(f"   ✅ Model validation PASSED! You are ready for deployment.")
        else:
            print(f"   ⚠️  Validation issues found. Check {report_path.name}")
    except Exception as e:
        print(f"   ⚠️  Validation skipped or errored: {e}")

    # ==================== 5. SUMMARY ====================
    print("\n" + "=" * 70)
    print("✨ BioImage.IO Export Complete!")
    print(f"📦 Final Ready-to-Upload Package: {zip_path}")
    print("=" * 70)
