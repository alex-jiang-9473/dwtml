"""Shared helpers for the split DWT SIREN pipeline."""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage

from siren import Siren

LEVELS = 2
WAVELET = "db4"
IMAGEID = "kodim08"
MODEL_DIR = f"results/dwt_siren_models/{IMAGEID}"
OUTPUT_DIR = "results/reconstructed_images"


def rgb_to_yuv(rgb_img):
    """Convert RGB to YUV (BT.601)."""
    rgb = np.array(rgb_img, dtype=np.float32)
    mat = np.array([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312],
    ])
    yuv = np.dot(rgb, mat.T)
    yuv[:, :, 1:] += 128.0
    return yuv


def yuv_to_rgb(yuv):
    """Convert YUV to RGB (BT.601)."""
    yuv_copy = yuv.copy()
    yuv_copy[:, :, 1:] -= 128.0
    mat = np.array([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0],
    ])
    rgb = np.dot(yuv_copy, mat.T)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def calculate_model_params(layers, hidden_size, dim_in=2, dim_out=1):
    """Calculate total parameters for a SIREN model."""
    first_layer = (dim_in * hidden_size) + hidden_size
    hidden_layers = (layers - 1) * ((hidden_size * hidden_size) + hidden_size)
    output_layer = (hidden_size * dim_out) + dim_out
    return first_layer + hidden_layers + output_layer


class EdgeHFNet(nn.Module):
    """Small CNN that predicts one HF band from LL and edge features."""

    def __init__(self, in_channels=3, hidden_channels=48, num_layers=5):
        super().__init__()
        num_layers = max(int(num_layers), 2)
        hidden_channels = int(hidden_channels)

        layers = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.GELU(),
            ])
        layers.append(nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_edge_cnn_inputs(ll_band):
    """Build LL normalization plus Sobel edge channels for CNN training/inference."""
    ll = np.asarray(ll_band, dtype=np.float32)
    ll_mean = float(np.mean(ll))
    ll_std = float(np.std(ll))
    if ll_std < 1e-8:
        ll_std = 1.0

    ll_norm = (ll - ll_mean) / ll_std
    sobel_y = ndimage.sobel(ll_norm, axis=0, mode="reflect").astype(np.float32)
    sobel_x = ndimage.sobel(ll_norm, axis=1, mode="reflect").astype(np.float32)

    inputs = np.stack([ll_norm, sobel_y, sobel_x], axis=0).astype(np.float32)
    stats = {
        "ll_mean": ll_mean,
        "ll_std": ll_std,
        "input_channels": 3,
        "feature_layout": {
            "type": "edge_cnn",
            "channels": ["ll_norm", "sobel_y", "sobel_x"],
        },
    }
    return inputs, stats


def find_model_size_for_budget(target_params, dim_in=2, dim_out=1, strict_under=True):
    """Find (layers, hidden_size) that fits a parameter budget."""
    best_config = (1, 3)
    best_params = calculate_model_params(1, 3, dim_in, dim_out)
    best_diff = abs(best_params - target_params)

    for layers in range(1, 100):
        hidden_size = 3 * layers
        params = calculate_model_params(layers, hidden_size, dim_in, dim_out)

        if strict_under and params > target_params:
            break

        diff = abs(params - target_params)
        if diff < best_diff:
            best_diff = diff
            best_config = (layers, hidden_size)
            best_params = params

        if params > target_params:
            break

    return best_config


def calculate_iterations_for_params(params, base_iterations=2000, iteration_factor=5, reference_params=10000):
    """Scale training iterations by parameter count."""
    iteration_scale = np.sqrt(params / reference_params)
    iterations = int(base_iterations * iteration_factor * iteration_scale)
    return max(500, min(iterations, base_iterations * iteration_factor * 2))


def make_norm_coords(coords_idx, h, w):
    """Normalize integer coordinates to [-1, 1]."""
    coords = coords_idx.astype(np.float32)
    coords[:, 0] = (coords[:, 0] / (h - 1)) * 2 - 1 if h > 1 else 0
    coords[:, 1] = (coords[:, 1] / (w - 1)) * 2 - 1 if w > 1 else 0
    return coords


def make_full_coords(h, w):
    """Build a full normalized coordinate grid for a band."""
    y_coords = np.linspace(-1, 1, h)
    x_coords = np.linspace(-1, 1, w)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
    return np.stack([yy.flatten(), xx.flatten()], axis=1).astype(np.float32)


def get_filter_threshold(channel_name: str, band_name: str, default_threshold: float = 1.5) -> float:
    """Resolve threshold factor using the experiment config module when available."""
    try:
        from experiment_config import get_filter_threshold as resolve_threshold

        return resolve_threshold(channel_name, band_name, default_threshold)
    except Exception:
        return default_threshold


def load_siren_checkpoint(model_path, device='cuda'):
    """Load a saved SIREN checkpoint and instantiate the model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    hidden_size = checkpoint.get('hidden_size', config.get('hidden_size'))
    layers = checkpoint.get('layers', config.get('layers'))
    dim_in = checkpoint.get('dim_in', config.get('dim_in', 2))
    dim_out = checkpoint.get('dim_out', config.get('dim_out', 1))
    w0 = checkpoint.get('w0', config.get('w0', 30.0))

    if hidden_size is None or layers is None:
        raise ValueError(f"Checkpoint at {model_path} is missing architecture metadata")

    model = Siren(
        dim_in=dim_in,
        dim_hidden=hidden_size,
        dim_out=dim_out,
        num_layers=layers,
        final_activation=None,
        w0_initial=w0,
        w0=w0,
    ).to(device)
    state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict'))
    if state_dict is None:
        raise ValueError(f"Checkpoint at {model_path} is missing state dict data")
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint
