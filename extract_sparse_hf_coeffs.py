"""Extract sparse HF DWT coefficients guided by edges from the original RGB image.

Saves per-band sparse metadata under `results/sparse_hf/{IMAGEID}`.
"""
import os
import sys
import json
import argparse
import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_config import IMAGEID, LEVELS, WAVELET
from dwt_siren_common import rgb_to_yuv


# Fraction of strongest image-edge pixels to keep per HF band (0.0-1.0)
# Set lower to keep fewer CNN-predicted pixels and make the HF bands sparser.
DEFAULT_EDGE_FRACTION = 0.05


def full_dwt_band(channel_coeffs, level_idx, band_name):
    band_index = {"cH": 0, "cV": 1, "cD": 2}[band_name]
    full_level_index = LEVELS - level_idx + 1
    return channel_coeffs[full_level_index][band_index]


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


class ImageEdgeCNN(nn.Module):
    """Small fixed-weight CNN that converts RGB into an edge saliency map."""

    def __init__(self):
        super().__init__()
        self.rgb_to_luma = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        with torch.no_grad():
            self.rgb_to_luma.weight.copy_(torch.tensor([[[[0.299]], [[0.587]], [[0.114]]]]))
            sobel_x = torch.tensor([[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]])
            sobel_y = torch.tensor([[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]])
            self.sobel_x.weight.copy_(sobel_x)
            self.sobel_y.weight.copy_(sobel_y)

        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, rgb_tensor):
        luma = self.rgb_to_luma(rgb_tensor)
        edge_x = self.sobel_x(luma)
        edge_y = self.sobel_y(luma)
        return torch.sqrt(edge_x * edge_x + edge_y * edge_y + 1e-12)


def build_image_edge_map(rgb_array):
    """Build an edge saliency map from the original RGB image using a tiny CNN."""
    rgb = np.asarray(rgb_array, dtype=np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    rgb_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)

    model = ImageEdgeCNN().eval()
    with torch.no_grad():
        edge_map = model(rgb_tensor).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

    # Emphasize strong edges so thresholding keeps more meaningful coordinates.
    edge_map = np.power(edge_map, 0.75, dtype=np.float32)

    edge_map = edge_map - float(edge_map.min())
    max_val = float(edge_map.max())
    if max_val > 1e-8:
        edge_map = edge_map / max_val
    return edge_map


def parse_args():
    parser = argparse.ArgumentParser(description="Extract CNN-guided sparse HF coefficients.")
    parser.add_argument(
        "--edge-fraction",
        type=float,
        default=DEFAULT_EDGE_FRACTION,
        help=f"Fraction of strongest edge pixels to keep per band (default: {DEFAULT_EDGE_FRACTION})",
    )
    return parser.parse_args()


def extract(edge_fraction):
    out_root = os.path.join("results", "sparse_hf", IMAGEID)
    ensure_dir(out_root)

    img_path = f"kodak-dataset/{IMAGEID}.png"
    img = Image.open(img_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    yuv = rgb_to_yuv(arr)
    channel_arrays = {
        "Y": yuv[:, :, 0],
        "U": yuv[:, :, 1],
        "V": yuv[:, :, 2],
    }

    manifest = {
        "image_id": IMAGEID,
        "levels": LEVELS,
        "wavelet": WAVELET,
        "edge_fraction": float(edge_fraction),
        "edge_source": "original_rgb_cnn",
        "bands": {},
    }

    image_edge_map = build_image_edge_map(arr)

    for channel_name in ["Y", "U", "V"]:
        channel_out = os.path.join(out_root, channel_name)
        ensure_dir(channel_out)

        coeffs = pywt.wavedec2(channel_arrays[channel_name], WAVELET, level=LEVELS)

        for level_idx in range(1, LEVELS + 1):
            level_out = os.path.join(channel_out, f"L{level_idx}")
            ensure_dir(level_out)

            band_edge_map = image_edge_map

            for band_name in ["cH", "cV", "cD"]:
                target = full_dwt_band(coeffs, level_idx, band_name)
                h, w = target.shape

                # Resize original-image edge map to the target subband size.
                if band_edge_map.shape != target.shape:
                    edge_tensor = torch.from_numpy(band_edge_map).unsqueeze(0).unsqueeze(0)
                    resized = F.interpolate(edge_tensor, size=target.shape, mode="bilinear", align_corners=False)
                    mag_rs = resized.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
                else:
                    mag_rs = band_edge_map

                if edge_fraction <= 0.0:
                    mask = np.zeros_like(mag_rs, dtype=bool)
                else:
                    thresh = np.percentile(mag_rs, 100.0 * (1.0 - edge_fraction))
                    mask = mag_rs >= thresh

                coords = np.argwhere(mask)
                values = target[mask]

                rel_path = os.path.join(channel_name, f"L{level_idx}", f"{channel_name}_{band_name}_L{level_idx}_sparse.pt")
                save_path = os.path.join(out_root, rel_path)
                metadata = {
                    "band_key": f"{channel_name}_{band_name}_L{level_idx}",
                    "shape": target.shape,
                    "sparse_coords": coords.astype(np.int32),
                    "sparse_values": values.astype(np.float32),
                    "fraction": float(edge_fraction),
                }
                # Save as torch file for convenience
                torch.save(metadata, save_path)

                manifest["bands"][metadata["band_key"]] = {
                    "sparse_metadata_path": save_path,
                    "num_sparse": int(coords.shape[0]),
                    "shape": list(target.shape),
                }

                print(f"Saved sparse band: {metadata['band_key']} | kept {coords.shape[0]} samples -> {save_path}")

    manifest_path = os.path.join(out_root, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nExtraction complete. Manifest: {manifest_path}")


if __name__ == "__main__":
    args = parse_args()
    extract(args.edge_fraction)
