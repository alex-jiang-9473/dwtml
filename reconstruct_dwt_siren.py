"""
Reconstruct images from trained SIREN models on DWT coefficients.
Loads models trained by train_dwt_siren.py and reconstructs full image.

Based on dwt_siren_split_yuv_channels.py logic.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import numpy as np
import pywt
import torch
from PIL import Image
import util
from experiment_config import IMAGEID, LEVELS, MODEL_DIR, OUTPUT_DIR, WAVELET
from dwt_siren_common import (
    load_siren_checkpoint,
    rgb_to_yuv,
    yuv_to_rgb,
)

# ============================================================================== 
# UTILITIES
# ============================================================================== 

def reconstruct_band_from_model(model, model_data, band_shape, sparse_mask=None, device='cuda'):
    """Reconstruct a single band using trained model
    
    Args:
        model: Trained SIREN model
        model_data: Dictionary with mean, std, shape
        band_shape: (h, w) of band
        sparse_mask: Optional mask for sparse reconstruction
        device: torch device
        
    Returns:
        Reconstructed coefficient array
    """
    h, w = band_shape
    
    if sparse_mask is not None:
        # Sparse reconstruction: only reconstruct at sparse locations
        sparse_coords_idx = np.argwhere(sparse_mask)
        
        # Normalize coordinates
        coords = sparse_coords_idx.astype(np.float32)
        coords[:, 0] = (coords[:, 0] / (h - 1)) * 2 - 1 if h > 1 else 0
        coords[:, 1] = (coords[:, 1] / (w - 1)) * 2 - 1 if w > 1 else 0
        
        # Predict
        coords_tensor = torch.FloatTensor(coords).to(device)
        with torch.no_grad():
            pred_norm = model(coords_tensor).cpu().numpy()
        
        # Denormalize
        pred = pred_norm * (model_data['coeff_std'] + 1e-8) + model_data['coeff_mean']
        
        # Fill result
        coeffs = np.zeros((h, w), dtype=np.float32)
        for idx, (y, x) in enumerate(sparse_coords_idx):
            if y < h and x < w:
                coeffs[y, x] = pred[idx, 0] if pred.ndim > 1 else pred[idx]
    else:
        # Dense reconstruction: reconstruct all pixels
        y_coords = np.linspace(-1, 1, h)
        x_coords = np.linspace(-1, 1, w)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
        coords = np.stack([yy.flatten(), xx.flatten()], axis=1).astype(np.float32)
        
        # Predict
        coords_tensor = torch.FloatTensor(coords).to(device)
        with torch.no_grad():
            pred_norm = model(coords_tensor).cpu().numpy()
        
        # Denormalize
        pred = pred_norm * (model_data['coeff_std'] + 1e-8) + model_data['coeff_mean']
        coeffs = pred.reshape(h, w)
    
    return coeffs

def load_training_manifest(model_dir):
    """Load the training manifest if it exists."""
    manifest_path = os.path.join(model_dir, 'manifest.json')
    if not os.path.exists(manifest_path):
        return None

    with open(manifest_path, 'r') as f:
        return json.load(f)


def reconstruct_channel_from_manifest(channel_name, original_coeffs, manifest, device='cuda'):
    """Reconstruct all bands for a channel from manifest-driven checkpoints."""
    coeffs_reconstructed = []
    channel_band_keys = manifest.get('channels', {}).get(channel_name, {}).get('bands', [])
    band_records = manifest.get('bands', {})

    def load_band_checkpoint(band_key):
        band_record = band_records.get(band_key)
        if not band_record:
            return None, None

        checkpoint_path = band_record.get('best_checkpoint')
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            return None, None

        model, checkpoint = load_siren_checkpoint(checkpoint_path, device)
        return model, checkpoint

    # LL band
    ll_key = f'{channel_name}_LL'
    ll_model, ll_checkpoint = load_band_checkpoint(ll_key)
    if ll_model is None:
        raise ValueError(f'LL band model not found for channel {channel_name}')

    ll_coeffs = reconstruct_band_from_model(
        ll_model,
        ll_checkpoint,
        tuple(ll_checkpoint['shape']),
        ll_checkpoint.get('sparse_mask'),
        device,
    )
    coeffs_reconstructed.append(ll_coeffs)

    # HF bands in PyWavelets order: [LL, level-2 details, level-1 details]
    for level_idx in range(1, LEVELS + 1):
        original_hf = original_coeffs[level_idx]
        hf_h, hf_w = original_hf[0].shape

        hf_bands = []
        for dir_name in ['cH', 'cV', 'cD']:
            band_key = f'{channel_name}_{dir_name}_L{level_idx}'
            if band_key in channel_band_keys:
                model, checkpoint = load_band_checkpoint(band_key)
                if model is not None:
                    band_shape = tuple(checkpoint['shape'])
                    hf_band = reconstruct_band_from_model(
                        model,
                        checkpoint,
                        band_shape,
                        checkpoint.get('sparse_mask'),
                        device,
                    )
                    hf_bands.append(hf_band)
                    continue

            hf_bands.append(np.zeros((hf_h, hf_w), dtype=np.float32))

        coeffs_reconstructed.append(tuple(hf_bands))

    return coeffs_reconstructed

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load original image for comparison
    img_path = f"kodak-dataset/{IMAGEID}.png"
    img_rgb = Image.open(img_path)
    print(f"\nLoaded original image: {img_path} ({img_rgb.size})")
    
    orig_h, orig_w = img_rgb.size[::-1]  # (H, W)
    
    # Convert to YUV
    img_yuv = rgb_to_yuv(img_rgb)
    y_channel = img_yuv[:, :, 0]
    u_channel = img_yuv[:, :, 1]
    v_channel = img_yuv[:, :, 2]
    
    # Perform DWT to get original coefficient shapes
    y_coeffs_orig = pywt.wavedec2(y_channel, WAVELET, level=LEVELS)
    u_coeffs_orig = pywt.wavedec2(u_channel, WAVELET, level=LEVELS)
    v_coeffs_orig = pywt.wavedec2(v_channel, WAVELET, level=LEVELS)
    
    print(f"\n{'='*70}")
    print("LOADING MODELS AND RECONSTRUCTING")
    print(f"{'='*70}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    manifest = load_training_manifest(MODEL_DIR)
    if manifest is None:
        raise FileNotFoundError(
            f"No manifest.json found in {MODEL_DIR}. Run train_dwt_siren.py first."
        )

    print(f"\nLoaded training manifest with {len(manifest.get('bands', {}))} trained bands")
    print(f"Compare configs: {manifest.get('compare_configs', False)}")
    print(f"HF training: {manifest.get('train_hf_bands', False)}")
    
    # Reconstruct channels
    print(f"\n{'='*70}")
    print("Reconstructing Y channel...")
    y_coeffs_rec = reconstruct_channel_from_manifest('Y', y_coeffs_orig, manifest, device)
    
    print(f"Reconstructing U channel...")
    u_coeffs_rec = reconstruct_channel_from_manifest('U', u_coeffs_orig, manifest, device)
    
    print(f"Reconstructing V channel...")
    v_coeffs_rec = reconstruct_channel_from_manifest('V', v_coeffs_orig, manifest, device)
    
    # Inverse DWT
    print(f"\nPerforming inverse DWT...")
    y_rec = pywt.waverec2(y_coeffs_rec, WAVELET)[:orig_h, :orig_w]
    u_rec = pywt.waverec2(u_coeffs_rec, WAVELET)[:orig_h, :orig_w]
    v_rec = pywt.waverec2(v_coeffs_rec, WAVELET)[:orig_h, :orig_w]
    
    # YUV to RGB
    yuv_rec = np.stack([y_rec, u_rec, v_rec], axis=2)
    rgb_rec = yuv_to_rgb(yuv_rec)
    
    # Save reconstructed image
    output_path = os.path.join(OUTPUT_DIR, f"{IMAGEID}_reconstructed.png")
    img_rec = Image.fromarray(rgb_rec)
    img_rec.save(output_path)
    print(f"\n✓ Saved to: {output_path}")
    
    # Calculate quality metrics
    y_psnr = util.get_clamped_psnr(
        torch.FloatTensor(y_channel.flatten()),
        torch.FloatTensor(y_rec.flatten())
    )
    u_psnr = util.get_clamped_psnr(
        torch.FloatTensor(u_channel.flatten()),
        torch.FloatTensor(u_rec.flatten())
    )
    v_psnr = util.get_clamped_psnr(
        torch.FloatTensor(v_channel.flatten()),
        torch.FloatTensor(v_rec.flatten())
    )
    rgb_psnr = util.get_clamped_psnr(
        torch.FloatTensor(np.array(img_rgb).flatten()),
        torch.FloatTensor(rgb_rec.flatten())
    )
    
    print(f"\n{'='*70}")
    print("RECONSTRUCTION QUALITY")
    print(f"{'='*70}")
    print(f"Y PSNR:  {y_psnr:.2f} dB")
    print(f"U PSNR:  {u_psnr:.2f} dB")
    print(f"V PSNR:  {v_psnr:.2f} dB")
    print(f"RGB PSNR: {rgb_psnr:.2f} dB")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
