import os
# Fix OpenMP duplicate library issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import random
import json
import time
import numpy as np
import pywt
import torch
from tqdm import tqdm
from PIL import Image
from training import Trainer
from siren import Siren
import util

# ---------------------------
# CONFIG
LEVELS = 2        # Number of DWT decomposition levels
WAVELET = "db4"   # Wavelet type
LOG_DIR = "results/dwt_split_yuv_channels"
IMAGEID = "kodim01"  # Image to compress

# Parameter budget from original RGB SIREN (10 layers, 28 hidden, 3 outputs)
# TOTAL_PARAM_BUDGET = 4346
TOTAL_PARAM_BUDGET = 60987

ITERA = 3000  # Training iterations per band model

# Parameter allocation percentages for YUV channels
Y_BUDGET_PERCENT = 0.6   # 70% for Y (luminance)
U_BUDGET_PERCENT = 0.2   # 15% for U (chrominance)
V_BUDGET_PERCENT = 0.2   # 15% for V (chrominance)

# HF bands parameter budget (as percentage of total budget)
# These are PART OF the channel budgets above, not additional
# LL budget = Channel budget - HF budget
Y_HF_BUDGET_PERCENT = 0.3   # 18% for Y channel HF bands (out of Y's 70%)
U_HF_BUDGET_PERCENT = 0.1   # 9% for U channel HF bands (out of U's 15%)
V_HF_BUDGET_PERCENT = 0.1   # 9% for V channel HF bands (out of V's 15%)

# Threshold for sparsity - higher = fewer HF coeffs = smaller models work better
THRESHOLD_FACTOR = 1.5  # Increased from 1.0 to keep only important coefficients

# Training options
SKIP_HF_TRAINING = False  # Set to True to only train LL bands (skip all HF bands)
USE_COMBINED_HF = False    # Combine cH, cV, cD into single 3-output model per level
USE_FP16 = True           # Use mixed precision (fp16) training for faster training and less memory

def rgb_to_yuv(rgb_img):
    """Convert RGB image to YUV color space (BT.601)"""
    rgb = np.array(rgb_img, dtype=np.float32)
    
    mat = np.array([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ])
    
    yuv = np.dot(rgb, mat.T)
    yuv[:, :, 1:] += 128.0
    
    return yuv

def yuv_to_rgb(yuv):
    """Convert YUV color space back to RGB (BT.601)"""
    yuv_copy = yuv.copy()
    yuv_copy[:, :, 1:] -= 128.0
    
    mat = np.array([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ])
    
    rgb = np.dot(yuv_copy, mat.T)
    rgb = np.clip(rgb, 0, 255)
    
    return rgb.astype(np.uint8)

def count_parameters(model):
    """Count total trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_model_params(layers, hidden_size, dim_in=2, dim_out=1):
    """Calculate total parameters for a SIREN model"""
    # First layer: (dim_in * hidden_size) + hidden_size
    first_layer = (dim_in * hidden_size) + hidden_size
    
    # Hidden layers: (layers - 1) * [(hidden_size * hidden_size) + hidden_size]
    hidden_layers = (layers - 1) * ((hidden_size * hidden_size) + hidden_size)
    
    # Output layer: (hidden_size * dim_out) + dim_out
    output_layer = (hidden_size * dim_out) + dim_out
    
    return first_layer + hidden_layers + output_layer

def find_model_size_for_budget(target_params, dim_in=2, dim_out=1):
    """Find (layers, hidden_size) that fits within parameter budget
    
    Constraint: hidden_size = 3 * layers (neurons are three times of layers)
    
    Returns (layers, hidden_size) that gets closest to target_params
    """
    best_config = (3, 9)
    best_diff = float('inf')
    
    # Search with constraint: hidden_size = 3 * layers
    for layers in range(3, 100):
        hidden_size = 3 * layers  # Enforce constraint
        params = calculate_model_params(layers, hidden_size, dim_in, dim_out)
        diff = abs(params - target_params)
        
        if diff < best_diff:
            best_diff = diff
            best_config = (layers, hidden_size)
        
        # If we're over budget, stop searching
        if params > target_params:
            break
    
    return best_config

def calculate_iterations_for_params(params, base_iterations=ITERA, reference_params=10000):
    """Calculate training iterations based on parameter count
    
    Larger models get more iterations, smaller models get fewer.
    Uses square root scaling to balance training time vs quality.
    
    Args:
        params: Number of parameters in the model
        base_iterations: Base iteration count (from ITERA config)
        reference_params: Reference parameter count for base_iterations
    
    Returns:
        Number of iterations (minimum 500, maximum 2x base_iterations)
    """
    # Square root scaling: larger models don't need proportionally more iterations
    scale = np.sqrt(params / reference_params)
    iterations = int(base_iterations * scale)
    
    # Clamp to reasonable range
    min_iters = 500
    max_iters = base_iterations * 2
    
    return max(min_iters, min(iterations, max_iters))

def allocate_parameters_per_channel(channel_coeffs, total_budget, hf_budget, channel_name):
    """Allocate parameter budget across DWT bands for a single channel
    
    Args:
        channel_coeffs: List of coefficient arrays [(LL), (cH, cV, cD), (cH, cV, cD), ...]
        total_budget: Parameters allocated for LL band in this channel
        hf_budget: Parameters allocated for HF bands in this channel
        channel_name: 'Y', 'U', or 'V'
    
    Returns:
        Dict mapping band names to (layers, hidden_size, allocated_params)
    """
    # Calculate importance scores based on coefficient count
    band_info = []
    
    # LL band - only Y channel trains ALL pixels, U/V use thresholding
    ll_coeffs = channel_coeffs[0]
    ll_energy = np.sum(ll_coeffs ** 2)
    
    if channel_name == 'Y':
        # Y channel LL: train ALL pixels
        ll_pixels = ll_coeffs.size
    else:
        # U/V channel LL: use threshold relative to std
        threshold = THRESHOLD_FACTOR * np.std(ll_coeffs)
        sparse_mask = np.abs(ll_coeffs) > threshold
        ll_pixels = np.sum(sparse_mask)
    
    if ll_pixels > 0:
        band_info.append(('LL', ll_coeffs, ll_energy, ll_pixels, 0))  # level=0 for LL
    
    # LL band allocation - use total_budget (LL budget)
    allocations = {}
    layers, hidden_size = find_model_size_for_budget(total_budget)
    actual_params = calculate_model_params(layers, hidden_size)
    
    allocations['LL'] = {
        'layers': layers,
        'hidden_size': hidden_size,
        'params': actual_params,
        'energy': ll_energy,
        'num_coeffs': ll_pixels
    }
    
    # If skipping HF training, return LL only
    if SKIP_HF_TRAINING:
        return allocations
    
    # HF bands at each level - only sparse coefficients above threshold
    for level_idx, (cH, cV, cD) in enumerate(channel_coeffs[1:], start=1):
        if USE_COMBINED_HF:
            # Combined mode: count all HF coefficients together at this level
            total_sparse = 0
            total_energy = 0
            
            for band_name, coeffs in [('cH', cH), ('cV', cV), ('cD', cD)]:
                threshold = THRESHOLD_FACTOR * np.std(coeffs)
                
                sparse_mask = np.abs(coeffs) > threshold
                num_sparse = np.sum(sparse_mask)
                total_sparse += num_sparse
                
                if num_sparse > 0:
                    total_energy += np.sum(coeffs[sparse_mask] ** 2)
            
            if total_sparse > 0:
                band_full_name = f"HF_L{level_idx}"  # Combined name
                # Use cH coeffs as placeholder for shape
                band_info.append((band_full_name, cH, total_energy, total_sparse, level_idx))
        else:
            # Separate mode: count sparse coefficients for each band
            for band_name, coeffs in [('cH', cH), ('cV', cV), ('cD', cD)]:
                # Threshold is adaptive: std() of UV coeffs is naturally smaller than Y
                threshold = THRESHOLD_FACTOR * np.std(coeffs)
                
                sparse_mask = np.abs(coeffs) > threshold
                num_sparse = np.sum(sparse_mask)
                
                if num_sparse > 0:
                    energy = np.sum(coeffs[sparse_mask] ** 2)
                    band_full_name = f"{band_name}_L{level_idx}"
                    band_info.append((band_full_name, coeffs, energy, num_sparse, level_idx))
    
    # Calculate allocation weights with importance scaling for HF bands only
    # LL band was already allocated above
    # Importance hierarchy: L1 HF > L2 HF
    weighted_coeffs = []
    for band_name, coeffs, energy, num_coeffs, level in band_info:
        if level == 0:  # Skip LL - already allocated
            continue
        elif level == 1:  # L1 HF bands - higher importance  
            importance = 1.5
        else:  # L2 HF bands
            importance = 0.8
        
        weighted_coeffs.append((band_name, coeffs, energy, num_coeffs, level, importance * num_coeffs))
    
    total_weighted = sum(info[5] for info in weighted_coeffs)
    
    remaining_budget = hf_budget
    
    # Allocate HF bands based on weighted coefficient count
    for i, (band_name, coeffs, energy, num_coeffs, level, weighted_count) in enumerate(weighted_coeffs):
        if i == len(weighted_coeffs) - 1:
            # Last band gets remaining budget
            allocated = remaining_budget
        else:
            # Allocate proportionally to weighted coefficient count
            weight = weighted_count / total_weighted if total_weighted > 0 else 1.0 / len(weighted_coeffs)
            allocated = int(hf_budget * weight)
            
            # Minimum for HF bands
            allocated = max(500, allocated)
        
        # Find best model size for this budget
        layers, hidden_size = find_model_size_for_budget(allocated)
        actual_params = calculate_model_params(layers, hidden_size)
        
        allocations[band_name] = {
            'layers': layers,
            'hidden_size': hidden_size,
            'params': actual_params,
            'energy': energy,
            'num_coeffs': num_coeffs
        }
        
        remaining_budget -= actual_params
    
    return allocations

def train_single_band_model(coeffs, coords, band_name, layers, hidden_size, device, iterations=1000, w0=30.0, dim_out=1):
    """Train a single-channel single-band SIREN model
    
    Args:
        coeffs: 1D or 2D array of coefficient values (N,) or (N, dim_out)
        coords: 2D array of normalized coordinates (N, 2)
        band_name: Name of the band (for logging)
        layers: Number of layers
        hidden_size: Hidden layer size
        device: torch device
        iterations: Training iterations
        w0: Omega_0 parameter for SIREN
        dim_out: Number of output channels (1 for single band, 3 for combined HF)
    
    Returns:
        Trained model, normalization params (mean, std), final PSNR
    """
    # Normalize coefficients - handle both 1D and 2D cases
    if coeffs.ndim == 1:
        coeffs = coeffs.reshape(-1, 1)
    
    coeff_mean = np.mean(coeffs, axis=0)
    coeff_std = np.std(coeffs, axis=0)
    
    # Skip normalization if std is too small (constant values)
    if np.any(coeff_std < 1e-6):
        print(f"  Warning: {band_name} has near-zero variance in some channels, using raw values")
        coeffs_norm = coeffs - coeff_mean
        coeff_std = np.where(coeff_std < 1e-6, 1.0, coeff_std)
    else:
        coeffs_norm = (coeffs - coeff_mean) / coeff_std
    
    # Convert to tensors
    coords_tensor = torch.FloatTensor(coords).to(device)
    coeffs_tensor = torch.FloatTensor(coeffs_norm).to(device)
    if coeffs_tensor.ndim == 1:
        coeffs_tensor = coeffs_tensor.unsqueeze(1)
    
    # Determine input dimension
    dim_in = coords.shape[1]
    
    # Create model
    model = Siren(
        dim_in=dim_in,
        dim_hidden=hidden_size,
        dim_out=dim_out,
        num_layers=layers,
        final_activation=None,
        w0_initial=w0,
        w0=w0
    ).to(device)
    
    # Adaptive learning rate based on band type and model size
    # Use more conservative LRs - too high causes divergence (loss=1)
    is_hf_band = band_name != 'LL'
    
    # if is_hf_band:
    #     # Conservative learning rates for HF bands (sparse data is sensitive)
    #     if hidden_size <= 24:
    #         lr = 2e-4
    #     elif hidden_size <= 40:
    #         lr = 1.5e-4
    #     elif hidden_size <= 56:
    #         lr = 1e-4
    #     else:
    #         lr = 8e-5
    # else:
    #     # Even more conservative for LL band (large dataset)
    #     if hidden_size <= 24:
    #         lr = 1e-4
    #     elif hidden_size <= 40:
    #         lr = 8e-5
    #     elif hidden_size <= 56:
    #         lr = 5e-5
    #     else:
    #         lr = 3e-5
    
    lr = 2e-4
    # Train with mixed precision if enabled
    if USE_FP16 and device.type == 'cuda':
        # Use automatic mixed precision
        scaler = torch.amp.GradScaler('cuda')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        model.train()
        pbar = tqdm(range(iterations), desc=f"Training {band_name}")
        for i in pbar:
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                pred = model(coords_tensor)
                loss = ((pred - coeffs_tensor) ** 2).mean()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update progress bar with loss
            if i % 10 == 0:
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    else:
        # Standard fp32 training
        trainer = Trainer(model, lr=lr)
        trainer.train(coords_tensor, coeffs_tensor, num_iters=iterations)
    
    # Calculate final PSNR
    model.eval()
    with torch.no_grad():
        pred_norm = model(coords_tensor).cpu().numpy()
        pred = pred_norm * (coeff_std + 1e-8) + coeff_mean
        true_flat = coeffs.flatten()
        pred_flat = pred.flatten()
        true_tensor = torch.FloatTensor(true_flat)
        pred_tensor = torch.FloatTensor(pred_flat)
        psnr = util.get_clamped_psnr(true_tensor, pred_tensor)
    
    return model, coeff_mean, coeff_std, psnr

def train_channel_dwt_models(channel_data, channel_coeffs, channel_name, param_budget, hf_budget, device):
    """Train all DWT band models for a single channel (Y, U, or V)
    
    Args:
        channel_data: Original channel data (H, W)
        channel_coeffs: DWT coefficients [(LL), (cH, cV, cD), ...]
        channel_name: 'Y', 'U', or 'V'
        param_budget: Parameters allocated for LL band in this channel
        hf_budget: Parameters allocated for HF bands in this channel
        device: torch device
    
    Returns:
        Dict of trained models and metadata
    """
    print(f"\n{'='*70}")
    print(f"Training {channel_name} Channel Models")
    print(f"  LL Band Budget: {param_budget:,}")
    print(f"  HF Bands Budget: {hf_budget:,}")
    print(f"  Total Budget: {param_budget + hf_budget:,}")
    print(f"{'='*70}")
    
    # Allocate parameters across bands
    allocations = allocate_parameters_per_channel(channel_coeffs, param_budget, hf_budget, channel_name)
    
    # Print allocation summary
    total_allocated = sum(info['params'] for info in allocations.values())
    print(f"\nParameter Allocation for {channel_name} Channel:")
    print(f"  {'Band':<12} {'Layers':<8} {'Hidden':<8} {'Params':<10} {'Energy':<12} {'Coeffs':<10}")
    print(f"  {'-'*70}")
    for band_name, info in allocations.items():
        print(f"  {band_name:<12} {info['layers']:<8} {info['hidden_size']:<8} "
              f"{info['params']:<10,} {info['energy']:<12.2f} {info['num_coeffs']:<10}")
    print(f"  {'-'*70}")
    print(f"  {'TOTAL':<12} {'':<8} {'':<8} {total_allocated:<10,}")
    print()
    
    models = {}
    
    # Train LL band
    ll_coeffs = channel_coeffs[0]
    h, w = ll_coeffs.shape
    
    ll_config = allocations['LL']
    
    if channel_name == 'Y':
        # Y channel LL: train ALL pixels
        y_coords = np.linspace(-1, 1, h)
        x_coords = np.linspace(-1, 1, w)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
        coords = np.stack([yy.flatten(), xx.flatten()], axis=1)
        
        coords_input = coords
        
        coeffs_to_train = ll_coeffs.flatten()
        num_train = ll_coeffs.size
        sparse_mask = None
        
        # More iterations for better convergence
        # if num_train > 20000:
        #     iterations = 5000
        # elif num_train > 10000:
        #     iterations = 6000
        # elif num_train > 5000:
        #     iterations = 7000
        # else:
        #     iterations = 8000
        iterations = calculate_iterations_for_params(ll_config['params'])
        print(f"Training LL band: {h}x{w} all pixels, {iterations} iterations")
    else:
        # U/V channel LL: use thresholding
        threshold = THRESHOLD_FACTOR * np.std(ll_coeffs)
        sparse_mask = np.abs(ll_coeffs) > threshold
        sparse_coords_idx = np.argwhere(sparse_mask)
        num_train = len(sparse_coords_idx)
        
        # Normalize sparse coordinates
        coords = sparse_coords_idx.astype(np.float32)
        coords[:, 0] = (coords[:, 0] / (h - 1)) * 2 - 1
        coords[:, 1] = (coords[:, 1] / (w - 1)) * 2 - 1
        

        coords_input = coords
        
        coeffs_to_train = ll_coeffs[sparse_mask]
        
        # Adaptive iterations based on model size
        iterations = calculate_iterations_for_params(ll_config['params'])
        sparsity_pct = 100.0 * num_train / (h * w)
        print(f"Training LL band: {num_train}/{h*w} coeffs ({sparsity_pct:.1f}%), {iterations} iterations")
    
    # LL band uses smaller w0 for smooth low-frequency content
    model_ll, ll_mean, ll_std, ll_psnr = train_single_band_model(
        coeffs_to_train, coords_input, 'LL',
        ll_config['layers'], ll_config['hidden_size'],
        device, iterations, w0=30.0, dim_out=1
    )
    
    models['LL'] = {
        'model': model_ll,
        'mean': ll_mean,
        'std': ll_std,
        'psnr': ll_psnr,
        'shape': (h, w),
        'sparse_mask': sparse_mask,  # None for Y, mask for U/V
        'num_coeffs': num_train,
        'layers': ll_config['layers'],
        'hidden_size': ll_config['hidden_size'],
        'params': ll_config['params']
    }
    
    print(f"  ✓ LL: PSNR={ll_psnr:.2f} dB, params={ll_config['params']:,}")
    
    # Train HF bands
    if SKIP_HF_TRAINING:
        print(f"\n  Skipping HF bands training (SKIP_HF_TRAINING=True)")
        return models
    
    if USE_COMBINED_HF:
        # Train combined HF bands (one 3-output model per level)
        for level_idx, (cH, cV, cD) in enumerate(channel_coeffs[1:], start=1):
            band_full_name = f"HF_L{level_idx}"
            
            if band_full_name not in allocations:
                continue
            
            h, w = cH.shape
            
            # Get sparse positions for all three bands
            # Threshold is adaptive: std() of UV coeffs is naturally smaller than Y
            
            # Union of all sparse positions across cH, cV, cD
            sparse_masks = []
            for coeffs in [cH, cV, cD]:
                thresh = THRESHOLD_FACTOR * np.std(coeffs)
                sparse_masks.append(np.abs(coeffs) > thresh)
            
            # Combine masks (union)
            combined_mask = np.logical_or.reduce(sparse_masks)
            sparse_coords_idx = np.argwhere(combined_mask)
            num_coeffs = len(sparse_coords_idx)
            
            if num_coeffs == 0:
                continue
            
            # Normalize coordinates
            coords_norm = sparse_coords_idx.astype(np.float32)
            coords_norm[:, 0] = (coords_norm[:, 0] / (h - 1)) * 2 - 1
            coords_norm[:, 1] = (coords_norm[:, 1] / (w - 1)) * 2 - 1
            

            coords_input = coords_norm
            
            # Get sparse coefficient values for all three bands
            sparse_coeffs = np.stack([
                cH[combined_mask],
                cV[combined_mask],
                cD[combined_mask]
            ], axis=1)  # Shape: (N, 3)
            
            config = allocations[band_full_name]
            
            iterations = calculate_iterations_for_params(config['params'])
            
            print(f"Training {band_full_name} (combined cH,cV,cD): {num_coeffs}/{h*w} coeffs, {iterations} iterations")
            model_hf, hf_mean, hf_std, hf_psnr = train_single_band_model(
                sparse_coeffs, coords_input, band_full_name,
                config['layers'], config['hidden_size'],
                device, iterations, w0=30.0, dim_out=3
            )
            
            models[band_full_name] = {
                'model': model_hf,
                'mean': hf_mean,
                'std': hf_std,
                'psnr': hf_psnr,
                'shape': (h, w),
                'sparse_mask': combined_mask,
                'num_coeffs': num_coeffs,
                'layers': config['layers'],
                'hidden_size': config['hidden_size'],
                'params': config['params'],
                'combined': True  # Flag to indicate this is a combined model
            }
            
            print(f"  ✓ {band_full_name}: PSNR={hf_psnr:.2f} dB, params={config['params']:,}")
        
        return models
    
    # Original separate HF band training
    for level_idx, (cH, cV, cD) in enumerate(channel_coeffs[1:], start=1):
        for band_name, coeffs in [('cH', cH), ('cV', cV), ('cD', cD)]:
            band_full_name = f"{band_name}_L{level_idx}"
            
            # Skip if not in allocations (no sparse coefficients)
            if band_full_name not in allocations:
                continue
            
            h, w = coeffs.shape
            
            # Get sparse positions
            threshold = THRESHOLD_FACTOR * np.std(coeffs)
            sparse_mask = np.abs(coeffs) > threshold
            sparse_coords_idx = np.argwhere(sparse_mask)
            num_coeffs = len(sparse_coords_idx)
            
            if num_coeffs == 0:
                continue
            
            # Normalize sparse coordinates
            coords_norm = sparse_coords_idx.astype(np.float32)
            coords_norm[:, 0] = (coords_norm[:, 0] / (h - 1)) * 2 - 1
            coords_norm[:, 1] = (coords_norm[:, 1] / (w - 1)) * 2 - 1
            

            coords_input = coords_norm
            
            # Get sparse coefficient values
            sparse_coeffs = coeffs[sparse_mask]
            
            config = allocations[band_full_name]
            
            iterations = calculate_iterations_for_params(config['params'])
            
            print(f"Training {band_full_name}: {num_coeffs}/{h*w} coeffs, {iterations} iterations")
            # Use standard w0=30.0 - too high causes saturation
            model_hf, hf_mean, hf_std, hf_psnr = train_single_band_model(
                sparse_coeffs, coords_input, band_full_name,
                config['layers'], config['hidden_size'],
                device, iterations, w0=30.0, dim_out=1
            )
            
            models[band_full_name] = {
                'model': model_hf,
                'mean': hf_mean,
                'std': hf_std,
                'psnr': hf_psnr,
                'shape': (h, w),
                'sparse_mask': sparse_mask,
                'num_coeffs': num_coeffs,
                'layers': config['layers'],
                'hidden_size': config['hidden_size'],
                'params': config['params']
            }
            
            print(f"  ✓ {band_full_name}: PSNR={hf_psnr:.2f} dB, params={config['params']:,}")
    
    return models

def train_combined_uv_dwt_models(u_channel, v_channel, u_coeffs, v_coeffs, param_budget, device):
    """Train combined UV models with 2-output models for U and V together
    
    This reduces model count by training one model per band with 2 outputs instead of separate U and V models
    """
    print(f"\n{'='*70}")
    print(f"Training Combined UV Channel Models")
    print(f"  Parameter Budget: {param_budget:,}")
    print(f"{'='*70}")
    
    # Use U channel structure for allocation (U and V have same structure)
    allocations = allocate_parameters_per_channel(u_coeffs, param_budget, 'U')
    
    # Update allocations for 2 outputs (U and V)
    for band_name, config in allocations.items():

        dim_in = 2
        actual_params = calculate_model_params(config['layers'], config['hidden_size'], dim_in, 2)
        config['params'] = actual_params
    
    total_allocated = sum(info['params'] for info in allocations.values())
    print(f"\nParameter Allocation for UV Channels:")
    print(f"  {'Band':<12} {'Layers':<8} {'Hidden':<8} {'Params':<10} {'Coeffs':<10}")
    print(f"  {'-'*70}")
    for band_name, info in allocations.items():
        print(f"  {band_name:<12} {info['layers']:<8} {info['hidden_size']:<8} "
              f"{info['params']:<10,} {info['num_coeffs']:<10}")
    print(f"  {'-'*70}")
    print(f"  {'TOTAL':<12} {'':<8} {'':<8} {total_allocated:<10,}")
    print()
    
    models = {}
    
    # Train LL band with 2 outputs (U and V combined)
    ll_u = u_coeffs[0]
    ll_v = v_coeffs[0]
    h, w = ll_u.shape
    
    ll_config = allocations['LL']
    
    # Use thresholding for UV LL bands
    threshold_u = THRESHOLD_FACTOR * np.std(ll_u)
    threshold_v = THRESHOLD_FACTOR * np.std(ll_v)
    sparse_mask_u = np.abs(ll_u) > threshold_u
    sparse_mask_v = np.abs(ll_v) > threshold_v
    # Union of masks
    sparse_mask = np.logical_or(sparse_mask_u, sparse_mask_v)
    sparse_coords_idx = np.argwhere(sparse_mask)
    num_train = len(sparse_coords_idx)
    
    # Normalize sparse coordinates
    coords = sparse_coords_idx.astype(np.float32)
    coords[:, 0] = (coords[:, 0] / (h - 1)) * 2 - 1
    coords[:, 1] = (coords[:, 1] / (w - 1)) * 2 - 1
    
    coords_input = coords
    
    # Stack U and V coefficients (N, 2)
    coeffs_to_train = np.stack([
        ll_u[sparse_mask],
        ll_v[sparse_mask]
    ], axis=1)
    
    # Adaptive iterations based on model size
    iterations = calculate_iterations_for_params(ll_config['params'])
    
    sparsity_pct = 100.0 * num_train / (h * w)
    print(f"Training UV LL band (2 outputs): {num_train}/{h*w} coeffs ({sparsity_pct:.1f}%), {iterations} iterations")
    
    model_ll, ll_mean, ll_std, ll_psnr = train_single_band_model(
        coeffs_to_train, coords_input, 'UV_LL',
        ll_config['layers'], ll_config['hidden_size'],
        device, iterations, w0=30.0, dim_out=2
    )
    
    models['LL'] = {
        'model': model_ll,
        'mean': ll_mean,  # Shape: (2,) for U and V
        'std': ll_std,    # Shape: (2,) for U and V
        'psnr': ll_psnr,
        'shape': (h, w),
        'sparse_mask': sparse_mask,
        'num_coeffs': num_train,
        'layers': ll_config['layers'],
        'hidden_size': ll_config['hidden_size'],
        'params': ll_config['params'],
        'combined_uv': True
    }
    
    print(f"  ✓ UV LL: PSNR={ll_psnr:.2f} dB, params={ll_config['params']:,}")
    
    # Skip HF for now
    if SKIP_HF_TRAINING:
        print(f"\n  Skipping HF bands training (SKIP_HF_TRAINING=True)")
    
    # Return models dict and separate U/V models for reconstruction
    # For reconstruction, we need to split the combined model outputs
    return models

def reconstruct_channel_from_models(models, channel_coeffs, device):
    """Reconstruct a channel by querying all trained models
    
    Args:
        models: Dict of trained models for each band
        channel_coeffs: Original DWT structure (for shapes)
        device: torch device
    
    Returns:
        Reconstructed channel coefficients ready for inverse DWT
    """
    reconstructed_coeffs = []
    
    # Reconstruct LL
    ll_info = models['LL']
    h, w = ll_info['shape']
    
    # Initialize with zeros
    ll_reconstructed = np.zeros((h, w), dtype=np.float32)
    
    if ll_info['sparse_mask'] is None:
        # Y channel: all pixels were trained
        y_coords = np.linspace(-1, 1, h)
        x_coords = np.linspace(-1, 1, w)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
        coords = np.stack([yy.flatten(), xx.flatten()], axis=1)
        
        coords_input = coords
        
        coords_tensor = torch.FloatTensor(coords_input).to(device)
        
        with torch.no_grad():
            pred_norm = ll_info['model'](coords_tensor).squeeze().cpu().numpy()
            pred = pred_norm * (ll_info['std'] + 1e-8) + ll_info['mean']
            ll_reconstructed = pred.reshape(h, w)
    else:
        # U/V channel: only sparse coefficients were trained
        sparse_mask = ll_info['sparse_mask']
        sparse_coords_idx = np.argwhere(sparse_mask)
        
        # Normalize coordinates
        coords_norm = sparse_coords_idx.astype(np.float32)
        coords_norm[:, 0] = (coords_norm[:, 0] / (h - 1)) * 2 - 1
        coords_norm[:, 1] = (coords_norm[:, 1] / (w - 1)) * 2 - 1
        
        coords_input = coords_norm
        
        coords_tensor = torch.FloatTensor(coords_input).to(device)
        
        with torch.no_grad():
            pred_norm = ll_info['model'](coords_tensor).squeeze().cpu().numpy()
            pred = pred_norm * (ll_info['std'] + 1e-8) + ll_info['mean']
        
        # Place predictions back at sparse positions
        ll_reconstructed[sparse_mask] = pred
    
    reconstructed_coeffs.append(ll_reconstructed)
    
    # Reconstruct HF bands for each level
    for level_idx, (cH_orig, cV_orig, cD_orig) in enumerate(channel_coeffs[1:], start=1):
        h, w = cH_orig.shape
        
        # Check if this level uses combined HF model
        combined_name = f"HF_L{level_idx}"
        if combined_name in models and models[combined_name].get('combined', False):
            # Combined reconstruction: one model outputs all 3 bands
            info = models[combined_name]
            sparse_mask = info['sparse_mask']
            sparse_coords_idx = np.argwhere(sparse_mask)
            
            # Normalize coordinates
            coords_norm = sparse_coords_idx.astype(np.float32)
            coords_norm[:, 0] = (coords_norm[:, 0] / (h - 1)) * 2 - 1
            coords_norm[:, 1] = (coords_norm[:, 1] / (w - 1)) * 2 - 1
            
            coords_tensor = torch.FloatTensor(coords_norm).to(device)
            
            with torch.no_grad():
                pred_norm = info['model'](coords_tensor).cpu().numpy()  # Shape: (N, 3)
                pred = pred_norm * (info['std'] + 1e-8) + info['mean']  # Broadcasting
            
            # Split predictions into three bands
            cH_reconstructed = np.zeros((h, w), dtype=np.float32)
            cV_reconstructed = np.zeros((h, w), dtype=np.float32)
            cD_reconstructed = np.zeros((h, w), dtype=np.float32)
            
            cH_reconstructed[sparse_mask] = pred[:, 0]
            cV_reconstructed[sparse_mask] = pred[:, 1]
            cD_reconstructed[sparse_mask] = pred[:, 2]
            
            reconstructed_level = [cH_reconstructed, cV_reconstructed, cD_reconstructed]
        else:
            # Separate reconstruction: individual models for each band
            reconstructed_level = []
            
            for band_name in ['cH', 'cV', 'cD']:
                band_full_name = f"{band_name}_L{level_idx}"
                
                # Initialize with zeros
                reconstructed_band = np.zeros((h, w), dtype=np.float32)
                
                if band_full_name in models:
                    info = models[band_full_name]
                    sparse_mask = info['sparse_mask']
                    sparse_coords_idx = np.argwhere(sparse_mask)
                    
                    # Normalize coordinates
                    coords_norm = sparse_coords_idx.astype(np.float32)
                    coords_norm[:, 0] = (coords_norm[:, 0] / (h - 1)) * 2 - 1
                    coords_norm[:, 1] = (coords_norm[:, 1] / (w - 1)) * 2 - 1
                    
                    coords_tensor = torch.FloatTensor(coords_norm).to(device)
                    
                    with torch.no_grad():
                        pred_norm = info['model'](coords_tensor).squeeze().cpu().numpy()
                        pred = pred_norm * (info['std'] + 1e-8) + info['mean']
                    
                    # Place predictions back
                    reconstructed_band[sparse_mask] = pred
                
                reconstructed_level.append(reconstructed_band)
        
        reconstructed_coeffs.append(tuple(reconstructed_level))
    
    return reconstructed_coeffs

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Load image
    img_path = f"kodak-dataset/{IMAGEID}.png"
    img_rgb = Image.open(img_path)
    print(f"\nLoaded image: {img_path}")
    print(f"  Size: {img_rgb.size}")
    
    # Convert to YUV
    img_yuv = rgb_to_yuv(img_rgb)
    y_channel = img_yuv[:, :, 0]
    u_channel = img_yuv[:, :, 1]
    v_channel = img_yuv[:, :, 2]
    
    print(f"\nYUV Statistics:")
    print(f"  Y: mean={np.mean(y_channel):.2f}, std={np.std(y_channel):.2f}, range=[{np.min(y_channel):.2f}, {np.max(y_channel):.2f}]")
    print(f"  U: mean={np.mean(u_channel):.2f}, std={np.std(u_channel):.2f}, range=[{np.min(u_channel):.2f}, {np.max(u_channel):.2f}]")
    print(f"  V: mean={np.mean(v_channel):.2f}, std={np.std(v_channel):.2f}, range=[{np.min(v_channel):.2f}, {np.max(v_channel):.2f}]")
    
    # Perform DWT on each channel
    print(f"\nPerforming {LEVELS}-level DWT with {WAVELET} wavelet on each channel...")
    y_coeffs = pywt.wavedec2(y_channel, WAVELET, level=LEVELS)
    u_coeffs = pywt.wavedec2(u_channel, WAVELET, level=LEVELS)
    v_coeffs = pywt.wavedec2(v_channel, WAVELET, level=LEVELS)
    
    # Parameter allocation strategy
    # Y gets more parameters (it carries more information)
    # U and V get equal share of the rest (chroma has less detail)
    # HF budgets are part of the channel budgets (subtracted from channel total)
    y_hf_budget = int(TOTAL_PARAM_BUDGET * Y_HF_BUDGET_PERCENT)
    u_hf_budget = int(TOTAL_PARAM_BUDGET * U_HF_BUDGET_PERCENT)
    v_hf_budget = int(TOTAL_PARAM_BUDGET * V_HF_BUDGET_PERCENT)
    
    y_budget = int(TOTAL_PARAM_BUDGET * Y_BUDGET_PERCENT) - y_hf_budget  # LL budget for Y
    u_budget = int(TOTAL_PARAM_BUDGET * U_BUDGET_PERCENT) - u_hf_budget  # LL budget for U
    v_budget = int(TOTAL_PARAM_BUDGET * V_BUDGET_PERCENT) - v_hf_budget  # LL budget for V
    
    y_ll_percent = (y_budget / TOTAL_PARAM_BUDGET) * 100
    u_ll_percent = (u_budget / TOTAL_PARAM_BUDGET) * 100
    v_ll_percent = (v_budget / TOTAL_PARAM_BUDGET) * 100
    
    print(f"\nParameter Budget Allocation:")
    print(f"  Total Budget: {TOTAL_PARAM_BUDGET:,} parameters")
    print(f"  Y Channel: {int(TOTAL_PARAM_BUDGET * Y_BUDGET_PERCENT):,} ({100*Y_BUDGET_PERCENT:.1f}%) = LL {y_budget:,} ({y_ll_percent:.1f}%) + HF {y_hf_budget:,} ({100*Y_HF_BUDGET_PERCENT:.1f}%)")
    print(f"  U Channel: {int(TOTAL_PARAM_BUDGET * U_BUDGET_PERCENT):,} ({100*U_BUDGET_PERCENT:.1f}%) = LL {u_budget:,} ({u_ll_percent:.1f}%) + HF {u_hf_budget:,} ({100*U_HF_BUDGET_PERCENT:.1f}%)")
    print(f"  V Channel: {int(TOTAL_PARAM_BUDGET * V_BUDGET_PERCENT):,} ({100*V_BUDGET_PERCENT:.1f}%) = LL {v_budget:,} ({v_ll_percent:.1f}%) + HF {v_hf_budget:,} ({100*V_HF_BUDGET_PERCENT:.1f}%)")
    
    # Train models for each channel separately
    start_time = time.time()
    
    y_models = train_channel_dwt_models(y_channel, y_coeffs, 'Y', y_budget, y_hf_budget, device)
    u_models = train_channel_dwt_models(u_channel, u_coeffs, 'U', u_budget, u_hf_budget, device)
    v_models = train_channel_dwt_models(v_channel, v_coeffs, 'V', v_budget, v_hf_budget, device)
    
    train_time = time.time() - start_time
    
    # Calculate actual total parameters
    actual_params_y = sum(info['params'] for info in y_models.values())
    actual_params_u = sum(info['params'] for info in u_models.values())
    actual_params_v = sum(info['params'] for info in v_models.values())
    total_actual_params = actual_params_y + actual_params_u + actual_params_v
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"  Total Time: {train_time:.2f} seconds")
    print(f"  Actual Total Parameters: {total_actual_params:,}")
    print(f"    Y: {actual_params_y:,}")
    print(f"    U: {actual_params_u:,}")
    print(f"    V: {actual_params_v:,}")
    print(f"  Budget Efficiency: {100*total_actual_params/TOTAL_PARAM_BUDGET:.1f}%")
    print(f"{'='*70}")
    
    # Save models and calculate size
    print("\nSaving models...")
    models_dir = os.path.join(LOG_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    total_model_size_bytes = 0
    
    # Save Y channel models
    for band_name, info in y_models.items():
        model_path = os.path.join(models_dir, f"y_{band_name.lower()}_model.pt")
        torch.save(info['model'].state_dict(), model_path)
        model_size = os.path.getsize(model_path)
        total_model_size_bytes += model_size
        print(f"  Y {band_name}: {model_size:,} bytes")
    
    # Save U channel models
    for band_name, info in u_models.items():
        model_path = os.path.join(models_dir, f"u_{band_name.lower()}_model.pt")
        torch.save(info['model'].state_dict(), model_path)
        model_size = os.path.getsize(model_path)
        total_model_size_bytes += model_size
        print(f"  U {band_name}: {model_size:,} bytes")
    
    # Save V channel models
    for band_name, info in v_models.items():
        model_path = os.path.join(models_dir, f"v_{band_name.lower()}_model.pt")
        torch.save(info['model'].state_dict(), model_path)
        model_size = os.path.getsize(model_path)
        total_model_size_bytes += model_size
        print(f"  V {band_name}: {model_size:,} bytes")
    
    total_model_size_kb = total_model_size_bytes / 1024
    total_model_size_mb = total_model_size_kb / 1024
    
    print(f"\n  Total Model Size: {total_model_size_bytes:,} bytes ({total_model_size_kb:.2f} KB, {total_model_size_mb:.2f} MB)")
    print(f"  Saved to: {models_dir}")
    
    # Reconstruct image
    print("\nReconstructing image...")
    y_reconstructed_coeffs = reconstruct_channel_from_models(y_models, y_coeffs, device)
    u_reconstructed_coeffs = reconstruct_channel_from_models(u_models, u_coeffs, device)
    v_reconstructed_coeffs = reconstruct_channel_from_models(v_models, v_coeffs, device)
    
    # Inverse DWT
    y_reconstructed = pywt.waverec2(y_reconstructed_coeffs, WAVELET)
    u_reconstructed = pywt.waverec2(u_reconstructed_coeffs, WAVELET)
    v_reconstructed = pywt.waverec2(v_reconstructed_coeffs, WAVELET)
    
    # Crop to original size if needed
    orig_h, orig_w = y_channel.shape
    y_reconstructed = y_reconstructed[:orig_h, :orig_w]
    u_reconstructed = u_reconstructed[:orig_h, :orig_w]
    v_reconstructed = v_reconstructed[:orig_h, :orig_w]
    
    # Combine channels and convert to RGB
    yuv_reconstructed = np.stack([y_reconstructed, u_reconstructed, v_reconstructed], axis=2)
    rgb_reconstructed = yuv_to_rgb(yuv_reconstructed)
    
    # Calculate PSNR per channel
    y_true = torch.FloatTensor(y_channel.flatten())
    y_pred = torch.FloatTensor(y_reconstructed.flatten())
    y_psnr = util.get_clamped_psnr(y_true, y_pred)
    
    u_true = torch.FloatTensor(u_channel.flatten())
    u_pred = torch.FloatTensor(u_reconstructed.flatten())
    u_psnr = util.get_clamped_psnr(u_true, u_pred)
    
    v_true = torch.FloatTensor(v_channel.flatten())
    v_pred = torch.FloatTensor(v_reconstructed.flatten())
    v_psnr = util.get_clamped_psnr(v_true, v_pred)
    
    # RGB PSNR
    rgb_true = torch.FloatTensor(np.array(img_rgb).flatten())
    rgb_pred = torch.FloatTensor(rgb_reconstructed.flatten())
    rgb_psnr = util.get_clamped_psnr(rgb_true, rgb_pred)
    
    print(f"\nFinal PSNR:")
    print(f"  Y Channel: {y_psnr:.2f} dB")
    print(f"  U Channel: {u_psnr:.2f} dB")
    print(f"  V Channel: {v_psnr:.2f} dB")
    print(f"  RGB Image: {rgb_psnr:.2f} dB")
    
    # Save reconstructed image
    output_path = os.path.join(LOG_DIR, f"{IMAGEID}_reconstructed.png")
    Image.fromarray(rgb_reconstructed).save(output_path)
    print(f"\nSaved reconstructed image to: {output_path}")
    
    # Save results
    results = {
        'image_id': IMAGEID,
        'total_params': total_actual_params,
        'params_y': actual_params_y,
        'params_u': actual_params_u,
        'params_v': actual_params_v,
        'param_budget': TOTAL_PARAM_BUDGET,
        'y_psnr': float(y_psnr),
        'u_psnr': float(u_psnr),
        'v_psnr': float(v_psnr),
        'rgb_psnr': float(rgb_psnr),
        'train_time': train_time,
        'levels': LEVELS,
        'wavelet': WAVELET,
        'total_model_size_bytes': total_model_size_bytes,
        'total_model_size_kb': round(total_model_size_kb, 2),
        'total_model_size_mb': round(total_model_size_mb, 2),
        'y_models': {k: {'layers': v['layers'], 'hidden_size': v['hidden_size'], 
                         'params': v['params'], 'psnr': float(v['psnr'])} 
                     for k, v in y_models.items()},
        'u_models': {k: {'layers': v['layers'], 'hidden_size': v['hidden_size'], 
                         'params': v['params'], 'psnr': float(v['psnr'])} 
                     for k, v in u_models.items()},
        'v_models': {k: {'layers': v['layers'], 'hidden_size': v['hidden_size'], 
                         'params': v['params'], 'psnr': float(v['psnr'])} 
                     for k, v in v_models.items()}
    }
    
    results_path = os.path.join(LOG_DIR, f"{IMAGEID}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to: {results_path}")

if __name__ == "__main__":
    main()
