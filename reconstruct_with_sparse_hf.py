"""Reconstruct image using trained LL SIREN models and CNN-extracted sparse HF coefficients.

This is a hybrid reconstruction path:
- LL bands come from the trained SIREN models.
- HF bands come from the sparse coefficient extraction driven by the image-edge CNN.
"""
import os
import sys
import json
import numpy as np
import pywt
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_config import IMAGEID, LEVELS, MODEL_DIR, OUTPUT_DIR, WAVELET
from reconstruct_dwt_siren import load_training_manifest, load_band_options
from dwt_siren_common import rgb_to_yuv, yuv_to_rgb
import util

try:
    from skimage.metrics import structural_similarity as ssim
except Exception:
    ssim = None


def compute_ssim_metric(original_rgb, reconstructed_rgb):
    """Compute RGB SSIM if scikit-image is available; otherwise return None."""
    if ssim is None:
        return None

    original = np.asarray(original_rgb, dtype=np.uint8)
    reconstructed = np.asarray(reconstructed_rgb, dtype=np.uint8)
    return float(ssim(original, reconstructed, channel_axis=2, data_range=255))


def load_sparse_metadata(root, channel_name, level_idx, band_name):
    path = os.path.join(root, channel_name, f"L{level_idx}", f"{channel_name}_{band_name}_L{level_idx}_sparse.pt")
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except Exception:
        return None


def build_sparse_hf_band(metadata, fallback_shape):
    shape = tuple(metadata.get('shape', fallback_shape))
    coeffs = np.zeros(shape, dtype=np.float32)
    coords = metadata.get('sparse_coords')
    values = metadata.get('sparse_values')
    if coords is None or values is None or len(coords) == 0:
        return coeffs

    coords = np.asarray(coords, dtype=int)
    values = np.asarray(values, dtype=np.float32)
    for i, (y, x) in enumerate(coords):
        if 0 <= y < shape[0] and 0 <= x < shape[1]:
            coeffs[y, x] = values[i]
    return coeffs


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    img_path = f"kodak-dataset/{IMAGEID}.png"
    img_rgb = Image.open(img_path)
    orig_h, orig_w = img_rgb.size[::-1]

    img_yuv = rgb_to_yuv(img_rgb)
    y_channel = img_yuv[:, :, 0]
    u_channel = img_yuv[:, :, 1]
    v_channel = img_yuv[:, :, 2]

    y_coeffs_orig = pywt.wavedec2(y_channel, WAVELET, level=LEVELS)
    u_coeffs_orig = pywt.wavedec2(u_channel, WAVELET, level=LEVELS)
    v_coeffs_orig = pywt.wavedec2(v_channel, WAVELET, level=LEVELS)

    print("Loading trained LL models from manifest...")
    manifest = load_training_manifest(MODEL_DIR)
    if manifest is None:
        raise FileNotFoundError(f"No manifest.json found in {MODEL_DIR}. Run train_dwt_siren.py first.")

    sparse_root = os.path.join('results', 'sparse_hf', IMAGEID)
    sparse_manifest_path = os.path.join(sparse_root, 'manifest.json')
    if not os.path.exists(sparse_manifest_path):
        raise FileNotFoundError(
            f"No sparse manifest found at {sparse_manifest_path}. Run extract_sparse_hf_coeffs.py first."
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Reconstruct LL bands using existing loader
    y_ll_options = load_band_options('Y', 'LL', y_coeffs_orig[0].shape, manifest, device)
    y_ll_coeffs = y_ll_options[0]['coeffs']

    u_ll_options = load_band_options('U', 'LL', u_coeffs_orig[0].shape, manifest, device)
    u_ll_coeffs = u_ll_options[0]['coeffs']

    v_ll_options = load_band_options('V', 'LL', v_coeffs_orig[0].shape, manifest, device)
    v_ll_coeffs = v_ll_options[0]['coeffs']

    def build_channel(channel_name, coeffs_orig):
        rec = [None]
        # LL
        if channel_name == 'Y':
            rec[0] = y_ll_coeffs
        elif channel_name == 'U':
            rec[0] = u_ll_coeffs
        else:
            rec[0] = v_ll_coeffs

        for level_idx in range(1, LEVELS + 1):
            extract_level_idx = LEVELS - level_idx + 1
            bands = []
            for band_name in ['cH', 'cV', 'cD']:
                metadata = load_sparse_metadata(sparse_root, channel_name, extract_level_idx, band_name)
                fallback_shape = coeffs_orig[level_idx][0].shape
                if metadata is None:
                    bands.append(np.zeros(fallback_shape, dtype=np.float32))
                else:
                    bands.append(build_sparse_hf_band(metadata, fallback_shape))
            rec.append(tuple(bands))
        return rec

    y_coeffs_rec = build_channel('Y', y_coeffs_orig)
    u_coeffs_rec = build_channel('U', u_coeffs_orig)
    v_coeffs_rec = build_channel('V', v_coeffs_orig)

    y_rec = pywt.waverec2(y_coeffs_rec, WAVELET)[:orig_h, :orig_w]
    u_rec = pywt.waverec2(u_coeffs_rec, WAVELET)[:orig_h, :orig_w]
    v_rec = pywt.waverec2(v_coeffs_rec, WAVELET)[:orig_h, :orig_w]

    yuv_rec = np.stack([y_rec, u_rec, v_rec], axis=2)
    rgb_rec = yuv_to_rgb(yuv_rec)

    original_rgb = np.asarray(img_rgb, dtype=np.uint8)
    rgb_psnr = float(util.get_clamped_psnr(
        torch.from_numpy(original_rgb.reshape(-1).astype(np.float32)),
        torch.from_numpy(rgb_rec.reshape(-1).astype(np.float32)),
    ))
    rgb_ssim = compute_ssim_metric(original_rgb, rgb_rec)

    out_path = os.path.join(OUTPUT_DIR, f"{IMAGEID}_reconstructed_ll_sparse_hf.png")
    Image.fromarray(rgb_rec).save(out_path)
    print(f"Saved reconstructed image: {out_path}")

    log_path = os.path.join(OUTPUT_DIR, f"{IMAGEID}_reconstructed_ll_sparse_hf.json")
    log_data = {
        "image_id": IMAGEID,
        "levels": LEVELS,
        "wavelet": WAVELET,
        "image_path": out_path,
        "metrics": {
            "rgb_psnr": rgb_psnr,
            "rgb_ssim": rgb_ssim,
        },
    }
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2)

    print(f"RGB PSNR: {rgb_psnr:.2f} dB")
    if rgb_ssim is not None:
        print(f"RGB SSIM: {rgb_ssim:.4f}")
    print(f"Saved metrics log: {log_path}")


if __name__ == '__main__':
    main()
