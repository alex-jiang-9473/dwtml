import os
import random
import json
import time
import numpy as np
import pywt
import torch
from PIL import Image
from training import Trainer
from siren import Siren
import util

# ---------------------------
# CONFIG
LEVELS = 2        # Number of DWT decomposition levels
WAVELET = "db4"   # Wavelet type
LOG_DIR = "results/dwt_2d_bands_yuv"
IMAGEID = "kodim01"  # Image to compress

# Model sizing parameters (same as grayscale version)
MODEL_SIZE_SCALE_LL = 0.8  # LL gets 64% of total parameters
MODEL_SIZE_SCALE_HF = 0.6  # Each HF gets 36% of total parameters
THRESHOLD_FACTOR = 1.0     # Sparsity threshold (1.0 * std)

def rgb_to_yuv(rgb_img):
    """Convert RGB image to YUV color space (BT.601)
    
    Note: Works with [0, 255] range to match grayscale version
    """
    rgb = np.array(rgb_img, dtype=np.float32)  # Keep [0, 255] range
    
    # BT.601 conversion matrix
    mat = np.array([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ])
    
    yuv = np.dot(rgb, mat.T)
    yuv[:, :, 1:] += 128.0  # U and V channels centered at 128 (like in [0,255] space)
    
    return yuv

def yuv_to_rgb(yuv):
    """Convert YUV color space back to RGB (BT.601)
    
    Note: Works with [0, 255] range
    """
    yuv_copy = yuv.copy()
    yuv_copy[:, :, 1:] -= 128.0  # Remove U/V centering
    
    # BT.601 inverse matrix
    mat = np.array([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ])
    
    rgb = np.dot(yuv_copy, mat.T)
    rgb = np.clip(rgb, 0, 255)
    
    return rgb.astype(np.uint8)

def get_adaptive_ll_lr(ll_coeffs):
    """Calculate adaptive learning rate for LL band (YUV version)
    
    Uses same formula as grayscale but applied to Y channel stats
    Reduces LR for larger models to prevent divergence
    """
    # Use Y channel statistics for LR calculation
    y_channel = ll_coeffs[:, :, 0]
    coeff_std = np.std(y_channel)
    coeff_range = np.max(y_channel) - np.min(y_channel)
    num_pixels = y_channel.size
    
    # Base LR depends on dataset size (aggressively reduced for large models)
    if num_pixels > 50000:
        base_lr = 8e-5   # Reduced from 1.2e-4
    elif num_pixels > 20000:
        base_lr = 1e-4   # Reduced from 1.5e-4
    elif num_pixels > 10000:
        base_lr = 1.2e-4  # Reduced from 2e-4
    elif num_pixels > 5000:
        base_lr = 1.5e-4  # Reduced from 2.5e-4
    else:
        base_lr = 2e-4   # Reduced from 3e-4
    
    # Variance factor (higher std = more variation = higher LR) - further reduced
    variance_factor = 1.0 + np.log10(max(coeff_std, 1.0)) / 7.0  # Reduced from /5.0
    
    # Range factor (wider range = more learning needed = higher LR) - further reduced
    range_factor = 1.0 + np.log10(max(coeff_range, 1.0)) / 8.0  # Reduced from /6.0
    
    # Size factor (fewer pixels = can use higher LR) - further reduced impact
    size_factor = 1.0 + (50000 - num_pixels) / 300000  # Reduced from /200000
    
    lr = base_lr * variance_factor * range_factor * size_factor
    lr = np.clip(lr, 5e-5, 2e-4)  # Reduced range: was [1e-4, 3e-4], now [5e-5, 2e-4]
    
    return lr

def get_adaptive_hf_lr(hf_coeffs, band_name, dwt_level):
    """Calculate adaptive learning rate for HF band (YUV version)
    
    Uses Y channel for sparse detection and statistics
    """
    # Use Y channel for statistics
    y_channel = hf_coeffs[:, :, 0]
    
    # Sparsity calculation
    threshold = THRESHOLD_FACTOR * np.std(y_channel)
    sparse_mask = np.abs(y_channel) > threshold
    num_coeffs = np.sum(sparse_mask)
    sparsity_ratio = num_coeffs / y_channel.size
    
    # Base LR depends on coefficient count (aggressively reduced)
    if num_coeffs > 10000:
        base_lr = 1.2e-4  # Reduced from 2e-4
    elif num_coeffs > 5000:
        base_lr = 8e-5    # Reduced from 1.2e-4
    else:
        base_lr = 2e-4    # Reduced from 3e-4
    
    # Count factor (fewer coeffs = higher LR) - further reduced
    if num_coeffs < 1000:
        count_factor = 1.2  # Reduced from 1.4
    elif num_coeffs < 5000:
        count_factor = 1.1  # Reduced from 1.2
    else:
        count_factor = 1.0
    
    # Variance factor - further reduced impact
    coeff_std = np.std(y_channel[sparse_mask]) if num_coeffs > 0 else 1.0
    variance_factor = 1.0 + np.log10(max(coeff_std, 1.0)) / 5.0  # Reduced from /4.0
    
    # Sparsity factor (sparser = fewer examples = higher LR) - further reduced impact
    sparsity_factor = 1.0 + (0.5 - sparsity_ratio) / 4.0  # Reduced from /3.0
    
    # Level factor (finer levels = more details = slightly higher LR) - further reduced
    level_factor = 1.0 + dwt_level * 0.03  # Reduced from 0.05
    
    lr = base_lr * count_factor * variance_factor * sparsity_factor * level_factor
    lr = np.clip(lr, 5e-5, 3e-4)  # Reduced max from 5e-4 to 3e-4
    
    return lr, num_coeffs

def get_model_size(ll_coeffs, hf_coeffs, band_name):
    """Calculate adaptive model size for LL or HF bands (YUV version)
    
    Returns (layers, hidden_size) based on coefficient count
    Uses Y channel for sizing decisions, but scales up for 3-channel output
    """
    import math
    
    if ll_coeffs is not None:
        # LL band sizing based on total pixels
        y_channel = ll_coeffs[:, :, 0]
        num_coeffs = y_channel.size
        
        log_coeffs = math.log10(max(num_coeffs, 10))
        
        # Layers: ~3 for 100 coeffs, ~5 for 500, ~8 for 2000, ~12 for 5000+
        layers = max(3, min(15, int(3.5 * log_coeffs - 4)))
        
        # Size: ~32 for 100, ~56 for 500, ~72 for 2000, ~96 for 5000+
        # Scale up by 1.2x for 3-channel output
        base_size = max(32, min(128, int(32 * log_coeffs - 32)))
        size = max(32, int(base_size * MODEL_SIZE_SCALE_LL * 1.2))
    else:
        # HF band sizing based on sparse Y coefficients
        y_channel = hf_coeffs[:, :, 0]
        threshold = THRESHOLD_FACTOR * np.std(y_channel)
        sparse_mask = np.abs(y_channel) > threshold
        num_coeffs = np.sum(sparse_mask)
        
        log_coeffs = math.log10(max(num_coeffs, 10))
        
        # Layers: ~3 for 100, ~5 for 500, ~8 for 2000, ~10 for 10k, ~13 for 30k+
        layers = max(3, min(15, int(3.5 * log_coeffs - 4)))
        
        # Size: ~32 for 100, ~48 for 500, ~64 for 2000, ~80 for 10k, ~96 for 30k+
        # Scale up by 1.3x for 3-channel output (HF needs more capacity than LL)
        base_size = max(32, min(128, int(32 * log_coeffs - 32)))
        size = max(32, int(base_size * MODEL_SIZE_SCALE_HF * 1.3))
    
    return layers, size

def train_ll_band(ll_coeffs_yuv, device):
    """Train a single LL model with 3 outputs (Y, U, V)
    
    Args:
        ll_coeffs_yuv: (H, W, 3) array with Y, U, V channels
        device: torch device
    
    Returns:
        Trained model, metrics dict
    """
    h, w, _ = ll_coeffs_yuv.shape
    
    # Normalize each channel separately
    ll_y_norm = ll_coeffs_yuv[:, :, 0]
    ll_u_norm = ll_coeffs_yuv[:, :, 1]
    ll_v_norm = ll_coeffs_yuv[:, :, 2]
    
    y_mean, y_std = np.mean(ll_y_norm), np.std(ll_y_norm)
    u_mean, u_std = np.mean(ll_u_norm), np.std(ll_u_norm)
    v_mean, v_std = np.mean(ll_v_norm), np.std(ll_v_norm)
    
    ll_y_norm = (ll_y_norm - y_mean) / (y_std + 1e-8)
    ll_u_norm = (ll_u_norm - u_mean) / (u_std + 1e-8)
    ll_v_norm = (ll_v_norm - v_mean) / (v_std + 1e-8)
    
    # Create coordinate grid [-1, 1] x [-1, 1]
    y_coords = np.linspace(-1, 1, h)
    x_coords = np.linspace(-1, 1, w)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
    coords = np.stack([yy.flatten(), xx.flatten()], axis=1)
    coords_tensor = torch.FloatTensor(coords).to(device)
    
    # Stack normalized coefficients (N, 3) - Y, U, V
    ll_tensor = torch.stack([
        torch.FloatTensor(ll_y_norm.flatten()),
        torch.FloatTensor(ll_u_norm.flatten()),
        torch.FloatTensor(ll_v_norm.flatten())
    ], dim=1).to(device)
    
    # Get model architecture
    layers, hidden_size = get_model_size(ll_coeffs_yuv, None, "LL")
    ll_lr = get_adaptive_ll_lr(ll_coeffs_yuv)
    
    # Adaptive iterations
    num_pixels = h * w
    if num_pixels < 5000:
        iterations = 3000
    elif num_pixels < 20000:
        iterations = 2000
    else:
        iterations = 1500
    
    print(f"\n{'='*60}")
    print(f"Training LL band (YUV) - {h}x{w} = {num_pixels} pixels")
    print(f"  Y: mean={y_mean:.3f}, std={y_std:.3f}")
    print(f"  U: mean={u_mean:.3f}, std={u_std:.3f}")
    print(f"  V: mean={v_mean:.3f}, std={v_std:.3f}")
    print(f"  Model: {layers} layers x {hidden_size} units")
    print(f"  LR: {ll_lr:.2e}, Iterations: {iterations}")
    print(f"{'='*60}")
    
    # Create model with 3 outputs
    model_ll = Siren(
        dim_in=2,
        dim_hidden=hidden_size,
        dim_out=3,  # Y, U, V channels
        num_layers=layers,
        final_activation=None,
        w0_initial=30.0,
        w0=30.0
    ).to(device)
    
    # Train
    trainer = Trainer(model_ll, lr=ll_lr)
    trainer.train(coords_tensor, ll_tensor, num_iters=iterations)
    
    # Calculate final PSNR per channel
    with torch.no_grad():
        pred = model_ll(coords_tensor)
        
        # Y channel PSNR
        y_pred_norm = pred[:, 0].cpu()
        y_pred = y_pred_norm * (y_std + 1e-8) + y_mean
        y_true = torch.FloatTensor(ll_coeffs_yuv[:, :, 0].flatten())
        y_psnr = util.get_clamped_psnr(y_true, y_pred)
        
        # U channel PSNR
        u_pred_norm = pred[:, 1].cpu()
        u_pred = u_pred_norm * (u_std + 1e-8) + u_mean
        u_true = torch.FloatTensor(ll_coeffs_yuv[:, :, 1].flatten())
        u_psnr = util.get_clamped_psnr(u_true, u_pred)
        
        # V channel PSNR
        v_pred_norm = pred[:, 2].cpu()
        v_pred = v_pred_norm * (v_std + 1e-8) + v_mean
        v_true = torch.FloatTensor(ll_coeffs_yuv[:, :, 2].flatten())
        v_psnr = util.get_clamped_psnr(v_true, v_pred)
    
    print(f"  Final: Y={y_psnr:.2f} dB, U={u_psnr:.2f} dB, V={v_psnr:.2f} dB")
    
    # Store normalization params
    metrics = {
        'y_mean': float(y_mean),
        'y_std': float(y_std),
        'u_mean': float(u_mean),
        'u_std': float(u_std),
        'v_mean': float(v_mean),
        'v_std': float(v_std),
        'y_psnr': float(y_psnr),
        'u_psnr': float(u_psnr),
        'v_psnr': float(v_psnr),
        'shape': (h, w),
        'num_pixels': num_pixels
    }
    
    return model_ll, metrics

def train_hf_band(hf_coeffs_yuv, band_name, dwt_level, device):
    """Train a single HF model with 3 outputs (Y, U, V)
    
    Uses Y channel to determine sparse positions, then trains all 3 channels
    at those same positions.
    
    Args:
        hf_coeffs_yuv: (H, W, 3) array with Y, U, V channels
        band_name: 'cH', 'cV', or 'cD'
        dwt_level: DWT decomposition level
        device: torch device
    
    Returns:
        Trained model, metrics dict
    """
    h, w, _ = hf_coeffs_yuv.shape
    
    # Use Y channel for sparse selection
    y_channel = hf_coeffs_yuv[:, :, 0]
    threshold = THRESHOLD_FACTOR * np.std(y_channel)
    sparse_mask = np.abs(y_channel) > threshold
    
    # Extract sparse positions from all 3 channels
    sparse_coords = np.argwhere(sparse_mask)
    num_coeffs = len(sparse_coords)
    
    if num_coeffs == 0:
        print(f"  {band_name}: No coefficients above threshold, skipping")
        return None, None
    
    # Normalize all 3 channels
    y_vals = y_channel[sparse_mask]
    u_vals = hf_coeffs_yuv[:, :, 1][sparse_mask]
    v_vals = hf_coeffs_yuv[:, :, 2][sparse_mask]
    
    y_mean, y_std = np.mean(y_vals), np.std(y_vals)
    u_mean, u_std = np.mean(u_vals), np.std(u_vals)
    v_mean, v_std = np.mean(v_vals), np.std(v_vals)
    
    y_norm = (y_vals - y_mean) / (y_std + 1e-8)
    u_norm = (u_vals - u_mean) / (u_std + 1e-8)
    v_norm = (v_vals - v_mean) / (v_std + 1e-8)
    
    # Create coordinate tensors
    coords_norm = sparse_coords.astype(np.float32)
    coords_norm[:, 0] = (coords_norm[:, 0] / (h - 1)) * 2 - 1
    coords_norm[:, 1] = (coords_norm[:, 1] / (w - 1)) * 2 - 1
    coords_tensor = torch.FloatTensor(coords_norm).to(device)
    
    # Stack coefficients (N, 3)
    coeffs_tensor = torch.stack([
        torch.FloatTensor(y_norm),
        torch.FloatTensor(u_norm),
        torch.FloatTensor(v_norm)
    ], dim=1).to(device)
    
    # Get model architecture and learning rate
    layers, hidden_size = get_model_size(None, hf_coeffs_yuv, band_name)
    hf_lr, _ = get_adaptive_hf_lr(hf_coeffs_yuv, band_name, dwt_level)
    
    # Adaptive iterations
    if num_coeffs < 1000:
        iterations = 3000
    elif num_coeffs < 5000:
        iterations = 2000
    else:
        iterations = 1500
    
    sparsity_pct = 100.0 * num_coeffs / (h * w)
    
    print(f"\n{'='*60}")
    print(f"Training {band_name} band (YUV) - Level {dwt_level}")
    print(f"  Shape: {h}x{w}, Sparse: {num_coeffs}/{h*w} ({sparsity_pct:.2f}%)")
    print(f"  Y: mean={y_mean:.3f}, std={y_std:.3f}")
    print(f"  U: mean={u_mean:.3f}, std={u_std:.3f}")
    print(f"  V: mean={v_mean:.3f}, std={v_std:.3f}")
    print(f"  Model: {layers} layers x {hidden_size} units")
    print(f"  LR: {hf_lr:.2e}, Iterations: {iterations}")
    print(f"{'='*60}")
    
    # Create model with 3 outputs
    model_hf = Siren(
        dim_in=2,
        dim_hidden=hidden_size,
        dim_out=3,  # Y, U, V channels
        num_layers=layers,
        final_activation=None,
        w0_initial=30.0,
        w0=30.0
    ).to(device)
    
    # Train
    trainer = Trainer(model_hf, lr=hf_lr)
    trainer.train(coords_tensor, coeffs_tensor, num_iters=iterations)
    
    # Calculate final metrics per channel
    with torch.no_grad():
        pred = model_hf(coords_tensor)
        
        # Y channel
        y_pred_norm = pred[:, 0].cpu()
        y_pred = y_pred_norm * (y_std + 1e-8) + y_mean
        y_true = torch.FloatTensor(y_vals)
        y_mse = torch.mean((y_pred - y_true) ** 2).item()
        y_mae = torch.mean(torch.abs(y_pred - y_true)).item()
        y_psnr = util.get_clamped_psnr(y_true, y_pred)
        
        # U channel
        u_pred_norm = pred[:, 1].cpu()
        u_pred = u_pred_norm * (u_std + 1e-8) + u_mean
        u_true = torch.FloatTensor(u_vals)
        u_mse = torch.mean((u_pred - u_true) ** 2).item()
        u_mae = torch.mean(torch.abs(u_pred - u_true)).item()
        u_psnr = util.get_clamped_psnr(u_true, u_pred)
        
        # V channel
        v_pred_norm = pred[:, 2].cpu()
        v_pred = v_pred_norm * (v_std + 1e-8) + v_mean
        v_true = torch.FloatTensor(v_vals)
        v_mse = torch.mean((v_pred - v_true) ** 2).item()
        v_mae = torch.mean(torch.abs(v_pred - v_true)).item()
        v_psnr = util.get_clamped_psnr(v_true, v_pred)
    
    print(f"  Final Y: PSNR={y_psnr:.2f} dB, MSE={y_mse:.2f}, MAE={y_mae:.2f}")
    print(f"  Final U: PSNR={u_psnr:.2f} dB, MSE={u_mse:.2f}, MAE={u_mae:.2f}")
    print(f"  Final V: PSNR={v_psnr:.2f} dB, MSE={v_mse:.2f}, MAE={v_mae:.2f}")
    
    # Store metrics
    metrics = {
        'y_mean': float(y_mean),
        'y_std': float(y_std),
        'u_mean': float(u_mean),
        'u_std': float(u_std),
        'v_mean': float(v_mean),
        'v_std': float(v_std),
        'y_psnr': float(y_psnr),
        'u_psnr': float(u_psnr),
        'v_psnr': float(v_psnr),
        'y_mse': float(y_mse),
        'u_mse': float(u_mse),
        'v_mse': float(v_mse),
        'y_mae': float(y_mae),
        'u_mae': float(u_mae),
        'v_mae': float(v_mae),
        'shape': (h, w),
        'num_coeffs': num_coeffs,
        'sparse_mask': sparse_mask,
        'threshold': float(threshold)
    }
    
    return model_hf, metrics

def reconstruct_progressive_image_multilevel(coeffs_yuv_list, metrics_all, device, output_prefix, img_shape):
    """Reconstruct image progressively for multi-level DWT
    
    Progressive stages go level by level, adding bands within each level
    """
    ll_yuv = coeffs_yuv_list[0]
    hf_bands_per_level = coeffs_yuv_list[1:]
    
    # Load original RGB
    img_rgb = Image.open(f"kodak-dataset/{IMAGEID}.png")
    img_rgb_array = np.array(img_rgb)
    h, w = img_shape
    
    progressive_results = []
    
    # Helper to reconstruct with specified bands
    def recon_bands(bands_dict, stage_name):
        # LL
        model_ll = metrics_all['ll']['model'].half()
        h_ll, w_ll, _ = ll_yuv.shape
        yy, xx = np.meshgrid(np.linspace(-1, 1, h_ll), np.linspace(-1, 1, w_ll), indexing='ij')
        coords_ll = torch.FloatTensor(np.stack([yy.flatten(), xx.flatten()], axis=1)).to(device).half()
        
        with torch.no_grad():
            pred_ll = model_ll(coords_ll).cpu().float().numpy()
        
        ll_recon = np.stack([
            (pred_ll[:, 0] * (metrics_all['ll']['y_std'] + 1e-8) + metrics_all['ll']['y_mean']).reshape(h_ll, w_ll),
            (pred_ll[:, 1] * (metrics_all['ll']['u_std'] + 1e-8) + metrics_all['ll']['u_mean']).reshape(h_ll, w_ll),
            (pred_ll[:, 2] * (metrics_all['ll']['v_std'] + 1e-8) + metrics_all['ll']['v_mean']).reshape(h_ll, w_ll)
        ], axis=2)
        
        # HF
        hf_recon_per_level = []
        total_size_kb = sum(p.numel() * 2 for p in metrics_all['ll']['model'].parameters()) / 1024
        
        for level_idx, (cH_orig, cV_orig, cD_orig) in enumerate(hf_bands_per_level, 1):
            level_hf = {}
            enabled_bands = bands_dict.get(level_idx, [])
            
            for band_name, band_orig in [('cH', cH_orig), ('cV', cV_orig), ('cD', cD_orig)]:
                key = f'level{level_idx}_{band_name}'
                if band_name in enabled_bands and key in metrics_all and metrics_all[key]['model'] is not None:
                    model_hf = metrics_all[key]['model'].half()
                    metrics_hf = metrics_all[key]
                    sparse_mask = metrics_hf['sparse_mask']
                    h_hf, w_hf = sparse_mask.shape
                    
                    sparse_coords = np.argwhere(sparse_mask).astype(np.float32)
                    sparse_coords[:, 0] = (sparse_coords[:, 0] / (h_hf - 1)) * 2 - 1
                    sparse_coords[:, 1] = (sparse_coords[:, 1] / (w_hf - 1)) * 2 - 1
                    coords_hf = torch.FloatTensor(sparse_coords).to(device).half()
                    
                    with torch.no_grad():
                        pred_hf = model_hf(coords_hf).cpu().float().numpy()
                    
                    band_recon = np.zeros((h_hf, w_hf, 3))
                    for c, (pred_c, mean, std) in enumerate([(pred_hf[:, 0], metrics_hf['y_mean'], metrics_hf['y_std']),
                                                               (pred_hf[:, 1], metrics_hf['u_mean'], metrics_hf['u_std']),
                                                               (pred_hf[:, 2], metrics_hf['v_mean'], metrics_hf['v_std'])]):
                        vals = pred_c * (std + 1e-8) + mean
                        band_recon[sparse_mask, c] = vals
                    
                    level_hf[band_name] = band_recon
                    total_size_kb += sum(p.numel() * 2 for p in metrics_all[key]['model'].parameters()) / 1024
                else:
                    level_hf[band_name] = np.zeros_like(band_orig)
            
            hf_recon_per_level.append((level_hf['cH'], level_hf['cV'], level_hf['cD']))
        
        # waverec2
        coeffs_list = [ll_recon] + hf_recon_per_level
        img_yuv_recon = np.zeros((h, w, 3))
        for c in range(3):
            coeffs_c = [coeffs_list[0][:, :, c]] + [(lb[0][:, :, c], lb[1][:, :, c], lb[2][:, :, c]) for lb in coeffs_list[1:]]
            img_yuv_recon[:, :, c] = pywt.waverec2(coeffs_c, WAVELET)
        
        img_rgb_recon = yuv_to_rgb(img_yuv_recon)
        psnr = util.get_clamped_psnr(torch.FloatTensor(img_rgb_array.flatten()), torch.FloatTensor(img_rgb_recon.flatten()))
        
        suffix = stage_name.lower().replace('+', '_').replace(':', '_').replace(' ', '_')
        Image.fromarray(img_rgb_recon).save(f"{output_prefix}_progressive_{suffix}.png")
        print(f"Progressive: {psnr:.2f} dB, {total_size_kb:.1f} kB → {output_prefix}_progressive_{suffix}.png")
        progressive_results.append((stage_name, psnr, total_size_kb))
    
    # LL only
    recon_bands({}, "LL")
    
    # Progressive: level by level, band by band
    num_levels = len(hf_bands_per_level)
    for level_idx in range(1, num_levels + 1):
        bands_dict = {prev: ['cH', 'cV', 'cD'] for prev in range(1, level_idx)}
        
        for bands in [['cH'], ['cH', 'cV'], ['cH', 'cV', 'cD']]:
            key = f'level{level_idx}_{bands[-1]}'
            if key in metrics_all and metrics_all[key]['model'] is not None:
                bands_dict[level_idx] = bands
                band_str = '+'.join(bands)
                recon_bands(bands_dict.copy(), f"LL+...+L{level_idx}:{band_str}")
    
    return progressive_results
    
    return progressive_results


def main():
    start_time = time.time()
    
    # Load RGB image
    img_rgb = Image.open(f"kodak-dataset/{IMAGEID}.png")
    img_rgb_array = np.array(img_rgb)
    
    print(f"Original RGB image: {img_rgb_array.shape}")
    
    # Convert RGB to YUV
    img_yuv = rgb_to_yuv(img_rgb)
    print(f"YUV image: {img_yuv.shape}")
    
    # Apply multi-level DWT to each YUV channel
    coeffs_y = pywt.wavedec2(img_yuv[:, :, 0], WAVELET, level=LEVELS)
    coeffs_u = pywt.wavedec2(img_yuv[:, :, 1], WAVELET, level=LEVELS)
    coeffs_v = pywt.wavedec2(img_yuv[:, :, 2], WAVELET, level=LEVELS)
    
    # Reorganize to multi-level format: [LL, (cH1,cV1,cD1), (cH2,cV2,cD2), ...]
    # Each band has shape (H, W, 3) for Y, U, V channels
    ll_yuv = np.stack([coeffs_y[0], coeffs_u[0], coeffs_v[0]], axis=2)
    
    # Store HF bands per level
    hf_bands_per_level = []
    for level_idx in range(1, len(coeffs_y)):
        cH_yuv = np.stack([coeffs_y[level_idx][0], coeffs_u[level_idx][0], coeffs_v[level_idx][0]], axis=2)
        cV_yuv = np.stack([coeffs_y[level_idx][1], coeffs_u[level_idx][1], coeffs_v[level_idx][1]], axis=2)
        cD_yuv = np.stack([coeffs_y[level_idx][2], coeffs_u[level_idx][2], coeffs_v[level_idx][2]], axis=2)
        hf_bands_per_level.append((cH_yuv, cV_yuv, cD_yuv))
    
    coeffs_yuv_combined = [ll_yuv] + hf_bands_per_level
    
    print(f"LL shape: {ll_yuv.shape}")
    for level_idx, (cH, cV, cD) in enumerate(hf_bands_per_level, 1):
        print(f"Level {level_idx}: cH={cH.shape}, cV={cV.shape}, cD={cD.shape}")
    
    #Calculate original size
    h, w = img_rgb_array.shape[:2]
    original_size_bytes = h * w * 3  # RGB, 8 bits per channel
    original_size_kb = original_size_bytes / 1024
    print(f"Original size: {original_size_kb:.2f} kB")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    seed = random.randint(1, int(1e6))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create output directory
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Train LL band
    model_ll, metrics_ll = train_ll_band(ll_yuv, device)
    
    # Collect all metrics and models
    metrics_all = {'ll': {**metrics_ll, 'model': model_ll}}
    
    # Train HF bands for each level
    for level_idx, (cH_yuv, cV_yuv, cD_yuv) in enumerate(hf_bands_per_level, 1):
        print(f"\n{'='*60}")
        print(f"Processing Level {level_idx}")
        print(f"{'='*60}")
        
        model_ch, metrics_ch = train_hf_band(cH_yuv, 'cH', dwt_level=level_idx, device=device)
        model_cv, metrics_cv = train_hf_band(cV_yuv, 'cV', dwt_level=level_idx, device=device)
        model_cd, metrics_cd = train_hf_band(cD_yuv, 'cD', dwt_level=level_idx, device=device)
        
        # Store with level-specific keys
        metrics_all[f'level{level_idx}_cH'] = {**metrics_ch, 'model': model_ch} if metrics_ch is not None else {'model': None}
        metrics_all[f'level{level_idx}_cV'] = {**metrics_cv, 'model': model_cv} if metrics_cv is not None else {'model': None}
        metrics_all[f'level{level_idx}_cD'] = {**metrics_cd, 'model': model_cd} if metrics_cd is not None else {'model': None}
    
    # Progressive reconstruction
    output_prefix = f"{LOG_DIR}/{IMAGEID}"
    progressive_results = reconstruct_progressive_image_multilevel(coeffs_yuv_combined, metrics_all, device, output_prefix, img_rgb_array.shape[:2])
    
    # Final reconstruction with all bands (FP16)
    print(f"\n{'='*60}")
    print("Final reconstruction (FP16)")
    print(f"{'='*60}")
    
    # Reconstruct LL
    model_ll_fp16 = model_ll.half()
    h_ll, w_ll, _ = ll_yuv.shape
    y_coords = np.linspace(-1, 1, h_ll)
    x_coords = np.linspace(-1, 1, w_ll)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
    coords_ll = np.stack([yy.flatten(), xx.flatten()], axis=1)
    coords_tensor_ll = torch.FloatTensor(coords_ll).to(device).half()
    
    with torch.no_grad():
        pred_ll = model_ll_fp16(coords_tensor_ll).cpu().float().numpy()
    
    y_ll = pred_ll[:, 0] * (metrics_ll['y_std'] + 1e-8) + metrics_ll['y_mean']
    u_ll = pred_ll[:, 1] * (metrics_ll['u_std'] + 1e-8) + metrics_ll['u_mean']
    v_ll = pred_ll[:, 2] * (metrics_ll['v_std'] + 1e-8) + metrics_ll['v_mean']
    
    ll_recon = np.stack([
        y_ll.reshape(h_ll, w_ll),
        u_ll.reshape(h_ll, w_ll),
        v_ll.reshape(h_ll, w_ll)
    ], axis=2)
    
    # Reconstruct HF bands for all levels
    hf_recon_per_level = []
    total_size_fp16 = sum(p.numel() * 2 for p in model_ll.parameters())
    
    for level_idx in range(1, LEVELS + 1):
        level_hf = {}
        for band_name in ['cH', 'cV', 'cD']:
            key = f'level{level_idx}_{band_name}'
            if key in metrics_all and metrics_all[key]['model'] is not None:
                model_hf = metrics_all[key]['model'].half()
                metrics_hf = metrics_all[key]
                sparse_mask = metrics_hf['sparse_mask']
                h_hf, w_hf = sparse_mask.shape
                
                sparse_coords = np.argwhere(sparse_mask)
                coords_norm = sparse_coords.astype(np.float32)
                coords_norm[:, 0] = (coords_norm[:, 0] / (h_hf - 1)) * 2 - 1
                coords_norm[:, 1] = (coords_norm[:, 1] / (w_hf - 1)) * 2 - 1
                coords_tensor_hf = torch.FloatTensor(coords_norm).to(device).half()
                
                with torch.no_grad():
                    pred_hf = model_hf(coords_tensor_hf).cpu().float().numpy()
                
                # Denormalize all 3 channels
                y_pred = pred_hf[:, 0] * (metrics_hf['y_std'] + 1e-8) + metrics_hf['y_mean']
                u_pred = pred_hf[:, 1] * (metrics_hf['u_std'] + 1e-8) + metrics_hf['u_mean']
                v_pred = pred_hf[:, 2] * (metrics_hf['v_std'] + 1e-8) + metrics_hf['v_mean']
                
                # Reconstruct full band
                band_recon = np.zeros((h_hf, w_hf, 3))
                for c, vals in enumerate([y_pred, u_pred, v_pred]):
                    band_recon[sparse_mask, c] = vals
                
                level_hf[band_name] = band_recon
                total_size_fp16 += sum(p.numel() * 2 for p in metrics_all[key]['model'].parameters())
            else:
                # Zero-filled if no model
                ref_shape = hf_bands_per_level[level_idx - 1][0].shape  # Use cH shape as reference
                level_hf[band_name] = np.zeros(ref_shape)
        
        hf_recon_per_level.append((level_hf['cH'], level_hf['cV'], level_hf['cD']))
    
    # Build coefficient list for waverec2: [LL, (cH1,cV1,cD1), (cH2,cV2,cD2), ...]
    coeffs_recon_list = [ll_recon] + hf_recon_per_level
    
    # Inverse DWT for each channel using waverec2
    img_yuv_recon = np.zeros((h, w, 3))
    for c in range(3):
        # Extract channel c from all bands
        coeffs_c = [coeffs_recon_list[0][:, :, c]]  # LL
        for level_bands in coeffs_recon_list[1:]:
            coeffs_c.append((level_bands[0][:, :, c], level_bands[1][:, :, c], level_bands[2][:, :, c]))
        recon_c = pywt.waverec2(coeffs_c, WAVELET)
        # Ensure correct size (crop if needed due to DWT boundary handling)
        img_yuv_recon[:, :, c] = recon_c[:h, :w]
    
    print(f"YUV reconstruction shape: {img_yuv_recon.shape}")
    print(f"YUV range: Y=[{img_yuv_recon[:,:,0].min():.1f}, {img_yuv_recon[:,:,0].max():.1f}], U=[{img_yuv_recon[:,:,1].min():.1f}, {img_yuv_recon[:,:,1].max():.1f}], V=[{img_yuv_recon[:,:,2].min():.1f}, {img_yuv_recon[:,:,2].max():.1f}]")
    
    # Convert to RGB
    img_rgb_recon = yuv_to_rgb(img_yuv_recon)
    print(f"RGB reconstruction shape: {img_rgb_recon.shape}, dtype: {img_rgb_recon.dtype}")
    print(f"RGB range: [{img_rgb_recon.min()}, {img_rgb_recon.max()}]")
    
    # Calculate final PSNR
    final_psnr = util.get_clamped_psnr(torch.FloatTensor(img_rgb_array.flatten()), torch.FloatTensor(img_rgb_recon.flatten()))
    
    # Calculate sizes and compression ratio
    total_size_kb = total_size_fp16 / 1024
    compression_ratio = original_size_kb / total_size_kb
    
    # Save final image
    Image.fromarray(img_rgb_recon).save(f"{output_prefix}_final.png")
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"PSNR: {final_psnr:.2f} dB")
    print(f"Model size (FP16): {total_size_kb:.1f} kB")
    print(f"Original size: {original_size_kb:.1f} kB")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Saved: {output_prefix}_final.png")
    
    # Save metrics
    metrics_json = {
        'image_id': IMAGEID,
        'image_shape': (h, w, 3),
        'wavelet': WAVELET,
        'levels': LEVELS,
        'final_psnr': float(final_psnr),
        'model_size_fp16_kb': float(total_size_kb),
        'original_size_kb': float(original_size_kb),
        'compression_ratio': float(compression_ratio),
        'll': {k: v for k, v in metrics_ll.items() if k != 'model'},
        'progressive_results': [{'stage': s, 'psnr': p, 'size_kb': sz} for s, p, sz in progressive_results],
        'training_time_sec': time.time() - start_time
    }
    
    # Add metrics for each level's HF bands
    for level_idx in range(1, LEVELS + 1):
        for band_name in ['cH', 'cV', 'cD']:
            key = f'level{level_idx}_{band_name}'
            if key in metrics_all and metrics_all[key].get('model') is not None:
                metrics_json[key] = {k: v for k, v in metrics_all[key].items() if k not in ['model', 'sparse_mask']}
    
    with open(f"{LOG_DIR}/metrics.json", 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    # Save models (FP16)
    torch.save(model_ll.half().state_dict(), f"{LOG_DIR}/best_model_ll.pt")
    for level_idx in range(1, LEVELS + 1):
        for band_name in ['cH', 'cV', 'cD']:
            key = f'level{level_idx}_{band_name}'
            if key in metrics_all and metrics_all[key].get('model') is not None:
                torch.save(metrics_all[key]['model'].half().state_dict(), f"{LOG_DIR}/best_model_level{level_idx}_{band_name}.pt")
    
    print(f"\nTotal time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
