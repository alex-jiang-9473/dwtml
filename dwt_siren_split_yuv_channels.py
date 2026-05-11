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
from resource_monitor import ResourceMonitor, print_memory_stats, reset_cuda_memory, get_memory_stats

# ---------------------------
# CONFIG
LEVELS = 2        # Number of DWT decomposition levels
WAVELET = "db4"   # Wavelet type
LOG_DIR = "results/dwt_split_yuv_channels"
IMAGEID = "kodim08"  # Image to compress

# Parameter budget from original RGB SIREN (10 layers, 28 hidden, 3 outputs)
# TOTAL_PARAM_BUDGET = 7479
# Parameter budget from original RGB SIREN (20 layers, 56 hidden, 3 outputs)
# TOTAL_PARAM_BUDGET = 60987
# Parameter budget from original RGB SIREN (30 layers, 86 hidden, 3 outputs)
TOTAL_PARAM_BUDGET = 217497

ITERA = 1000  # Single training-iteration knob; increase/decrease to scale all band training

# Parameter allocation percentages for YUV channels
Y_BUDGET_PERCENT = 0.4   # 60% of total budget for Y (luminance)
U_BUDGET_PERCENT = 0.3   # 20% of total budget for U (chrominance)
V_BUDGET_PERCENT = 0.3   # 20% of total budget for V (chrominance)

# HF bands parameter budget as a fraction of each channel's own budget.
# LL budget = Channel budget - HF budget.
Y_HF_BUDGET_PERCENT = 0.7   # 30% of Y channel budget for HF bands
U_HF_BUDGET_PERCENT = 0.7   # 30% of U channel budget for HF bands
V_HF_BUDGET_PERCENT = 0.7   # 30% of V channel budget for HF bands

# Threshold for sparsity - higher = fewer HF coeffs = smaller models work better
THRESHOLD_FACTOR = 1  # Increased from 1.0 to keep only important coefficients

# Y-LL band sparsity: 0.5 = keep top 50% of coefficients by magnitude, 1.0 = keep all
Y_LL_SALIENT_FRACTION = 1  # Keep top N% of Y-LL coefficients (0.0 to 1.0)

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

def find_model_size_for_budget(target_params, dim_in=2, dim_out=1, strict_under=True):
    """Find (layers, hidden_size) that fits within parameter budget
    
    Constraint: hidden_size = 3 * layers (neurons are three times of layers)
    
    Args:
        target_params: Target parameter count
        dim_in: Input dimension
        dim_out: Output dimension
        strict_under: If True, only return configs UNDER target_params
    
    Returns (layers, hidden_size) that gets closest to target_params without exceeding it
    """
    best_config = (1, 3)  # Start with minimum size (1 layer, hidden_size=3)
    best_params = calculate_model_params(1, 3, dim_in, dim_out)
    best_diff = abs(best_params - target_params)
    
    # Search with constraint: hidden_size = 3 * layers
    for layers in range(1, 100):
        hidden_size = 3 * layers  # Enforce constraint
        params = calculate_model_params(layers, hidden_size, dim_in, dim_out)
        
        # If strict, only consider configs under budget
        if strict_under and params > target_params:
            break
        
        diff = abs(params - target_params)
        if diff < best_diff:
            best_diff = diff
            best_config = (layers, hidden_size)
            best_params = params
        
        # If we're over budget and not strict, stop searching
        if params > target_params:
            break
    
    return best_config

def calculate_iterations_for_params(params, base_iterations=ITERA):
    """Calculate training iterations based on parameter count
    
    Larger models get more iterations, smaller models get fewer.
    Uses square root scaling to balance training time vs quality.
    
    Args:
        params: Number of parameters in the model
        base_iterations: Base iteration count (from ITERA config)
    
    Returns:
        Number of iterations (clamped to a fixed fraction/multiple of base_iterations)
    """
    # Square root scaling: larger models don't need proportionally more iterations
    reference_params = max(base_iterations, 1)
    scale = np.sqrt(params / reference_params)
    iterations = int(base_iterations * scale)
    
    # Clamp to a fixed range derived from the same knob
    min_iters = max(500, int(base_iterations * 0.25))
    max_iters = int(base_iterations * 2)
    
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
    
    # LL band - Y channel uses salient filtering, U/V use thresholding
    ll_coeffs = channel_coeffs[0]
    
    if channel_name == 'Y':
        # Y channel LL: keep top N% of coefficients by magnitude (salient filtering)
        if Y_LL_SALIENT_FRACTION < 1.0:
            # Keep coefficients above the (1 - Y_LL_SALIENT_FRACTION) quantile
            threshold = np.quantile(np.abs(ll_coeffs), 1.0 - Y_LL_SALIENT_FRACTION)
            sparse_mask = np.abs(ll_coeffs) > threshold
            ll_pixels = np.sum(sparse_mask)
        else:
            # Keep all pixels if Y_LL_SALIENT_FRACTION == 1.0
            ll_pixels = ll_coeffs.size
    else:
        # U/V channel LL: use threshold relative to std
        threshold = THRESHOLD_FACTOR * np.std(ll_coeffs)
        sparse_mask = np.abs(ll_coeffs) > threshold
        ll_pixels = np.sum(sparse_mask)
    
    if ll_pixels > 0:
        band_info.append(('LL', ll_coeffs, ll_pixels, 0))  # level=0 for LL
    
    # LL band allocation - use total_budget (LL budget) with strict enforcement
    allocations = {}
    layers, hidden_size = find_model_size_for_budget(total_budget, dim_in=2, dim_out=1, strict_under=True)
    actual_params = calculate_model_params(layers, hidden_size, dim_in=2, dim_out=1)
    
    allocations['LL'] = {
        'layers': layers,
        'hidden_size': hidden_size,
        'params': actual_params,
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
            
            for band_name, coeffs in [('cH', cH), ('cV', cV), ('cD', cD)]:
                threshold = THRESHOLD_FACTOR * np.std(coeffs)
                
                sparse_mask = np.abs(coeffs) > threshold
                num_sparse = np.sum(sparse_mask)
                total_sparse += num_sparse
            
            if total_sparse > 0:
                band_full_name = f"HF_L{level_idx}"  # Combined name
                # Use cH coeffs as placeholder for shape
                band_info.append((band_full_name, cH, total_sparse, level_idx))
        else:
            # Separate mode: count sparse coefficients for each band
            for band_name, coeffs in [('cH', cH), ('cV', cV), ('cD', cD)]:
                # Threshold is adaptive: std() of UV coeffs is naturally smaller than Y
                threshold = THRESHOLD_FACTOR * np.std(coeffs)
                
                sparse_mask = np.abs(coeffs) > threshold
                num_sparse = np.sum(sparse_mask)
                
                if num_sparse > 0:
                    band_full_name = f"{band_name}_L{level_idx}"
                    band_info.append((band_full_name, coeffs, num_sparse, level_idx))
    
    # Calculate allocation weights with importance scaling for HF bands only
    # LL band was already allocated above
    # Importance hierarchy: L1 HF > L2 HF
    weighted_coeffs = []
    for band_name, coeffs, num_coeffs, level in band_info:
        if level == 0:  # Skip LL - already allocated
            continue
        elif level == 1:  # L1 HF bands - higher importance  
            importance = 1.5
        else:  # L2 HF bands
            importance = 0.8
        
        weighted_coeffs.append((band_name, coeffs, num_coeffs, level, importance * num_coeffs))
    
    # If no HF bands or HF budget too small, skip
    if not weighted_coeffs or hf_budget < 50:
        return allocations
    
    total_weighted = sum(info[4] for info in weighted_coeffs)
    
    # Minimum viable model size (1 layer, 3 hidden)
    min_model_params = calculate_model_params(1, 3, dim_in=2, dim_out=1)
    
    # First pass: determine which bands can be allocated
    viable_bands = []
    for band_name, coeffs, num_coeffs, level, weighted_count in weighted_coeffs:
        weight = weighted_count / total_weighted if total_weighted > 0 else 1.0 / len(weighted_coeffs)
        initial_allocation = int(hf_budget * weight)
        
        # Check if this band can get minimum viable model
        if initial_allocation >= min_model_params:
            viable_bands.append((band_name, coeffs, num_coeffs, level, weighted_count, weight))
    
    # If no viable bands, skip HF allocation
    if not viable_bands:
        return allocations
    
    # Second pass: allocate to viable bands with strict budget enforcement
    remaining_budget = hf_budget
    for i, (band_name, coeffs, num_coeffs, level, weighted_count, weight) in enumerate(viable_bands):
        if i == len(viable_bands) - 1:
            # Last band gets all remaining budget
            allocated = remaining_budget
        else:
            # Allocate proportionally
            allocated = int(hf_budget * weight)
        
        # Make sure we don't exceed remaining budget
        allocated = min(allocated, remaining_budget)
        
        if allocated < min_model_params:
            # Skip if not enough budget
            continue
        
        # Find best model size UNDER this budget
        layers, hidden_size = find_model_size_for_budget(allocated, dim_in=2, dim_out=1, strict_under=True)
        actual_params = calculate_model_params(layers, hidden_size, dim_in=2, dim_out=1)
        
        allocations[band_name] = {
            'layers': layers,
            'hidden_size': hidden_size,
            'params': actual_params,
            'num_coeffs': num_coeffs
        }
        
        remaining_budget -= actual_params
        
        # Safety check: stop if we run out of budget
        if remaining_budget < min_model_params:
            break
    
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

def train_channel_dwt_models(channel_data, channel_coeffs, channel_name, param_budget, hf_budget, device, train_only_ll=False, train_only_level=None, train_only_band=None):
    """Train all DWT band models for a single channel (Y, U, or V)
    
    Args:
        channel_data: Original channel data (H, W)
        channel_coeffs: DWT coefficients [(LL), (cH, cV, cD), ...]
        channel_name: 'Y', 'U', or 'V'
        param_budget: Parameters allocated for LL band in this channel
        hf_budget: Parameters allocated for HF bands in this channel
        device: torch device
        train_only_ll: If True, only train LL band and return
        train_only_level: If set (1 or 2), only train HF bands for that level
        train_only_band: If set (e.g., 'LL', 'cH_L1'), only train that specific band
    
    Returns:
        Dict of trained models and metadata
    """
    # Skip allocation if only training specific level or band
    if train_only_band is not None:
        print(f"Training {channel_name} Channel - {train_only_band} Band Only")
    elif train_only_level is not None:
        print(f"\n{'='*70}")
        print(f"Training {channel_name} Channel - Level {train_only_level} HF Bands Only")
        print(f"{'='*70}")
    elif train_only_ll:
        print(f"\n{'='*70}")
        print(f"Training {channel_name} Channel - LL Band Only")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print(f"Training {channel_name} Channel Models")
        print(f"  LL Band Budget: {param_budget:,}")
        print(f"  HF Bands Budget: {hf_budget:,}")
        print(f"  Total Budget: {param_budget + hf_budget:,}")
        print(f"{'='*70}")
    
    # Allocate parameters across bands (needed for all cases to know configuration)
    allocations = allocate_parameters_per_channel(channel_coeffs, param_budget, hf_budget, channel_name)
    
    # Print allocation summary only if training all
    if not train_only_ll and train_only_level is None:
        total_allocated = sum(info['params'] for info in allocations.values())
        print(f"\nParameter Allocation for {channel_name} Channel:")
        print(f"  {'Band':<12} {'Layers':<8} {'Hidden':<8} {'Params':<10} {'Coeffs':<10}")
        print(f"  {'-'*60}")
        for band_name, info in allocations.items():
            print(f"  {band_name:<12} {info['layers']:<8} {info['hidden_size']:<8} "
                  f"{info['params']:<10,} {info['num_coeffs']:<10}")
        print(f"  {'-'*60}")
        print(f"  {'TOTAL':<12} {'':<8} {'':<8} {total_allocated:<10,}")
        print()
    
    models = {}
    
    # Train LL band (if not training only specific HF level or if training only LL)
    if train_only_level is None and (train_only_band is None or train_only_band == 'LL'):
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
        
        # Return early if only training LL band
        if train_only_ll or train_only_band == 'LL':
            return models
    
    # Train HF bands
    if SKIP_HF_TRAINING:
        print(f"\n  Skipping HF bands training (SKIP_HF_TRAINING=True)")
        return models
    
    # Skip LL if only training specific HF level
    if train_only_level is not None:
        pass  # Don't return, continue to HF training
    
    if USE_COMBINED_HF:
        # Train combined HF bands (one 3-output model per level)
        for level_idx, (cH, cV, cD) in enumerate(channel_coeffs[1:], start=1):
            # Skip if only training specific level or band
            if train_only_level is not None and level_idx != train_only_level:
                continue
                
            band_full_name = f"HF_L{level_idx}"
            
            # Skip if training only specific band and this isn't it
            if train_only_band is not None and train_only_band != band_full_name:
                continue
            
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
        # Skip if only training specific level
        if train_only_level is not None and level_idx != train_only_level:
            continue
            
        for band_name, coeffs in [('cH', cH), ('cV', cV), ('cD', cD)]:
            band_full_name = f"{band_name}_L{level_idx}"
            
            # Skip if training only specific band and this isn't it
            if train_only_band is not None and train_only_band != band_full_name:
                continue
            
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
    
    # Initialize resource monitoring
    monitor = ResourceMonitor(device='cuda' if torch.cuda.is_available() else 'cpu')
    monitor.start()
    print_memory_stats("Memory at script start", device='cuda' if torch.cuda.is_available() else 'cpu')
    
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
    print_memory_stats("Memory before DWT", device='cuda' if torch.cuda.is_available() else 'cpu')
    
    y_coeffs = pywt.wavedec2(y_channel, WAVELET, level=LEVELS)
    u_coeffs = pywt.wavedec2(u_channel, WAVELET, level=LEVELS)
    v_coeffs = pywt.wavedec2(v_channel, WAVELET, level=LEVELS)
    
    # Parameter allocation strategy
    # Y gets more parameters (it carries more information)
    # U and V get equal share of the rest (chroma has less detail)
    # HF budgets are fractions of each channel's own budget, not the full total.

    y_channel_budget = int(TOTAL_PARAM_BUDGET * Y_BUDGET_PERCENT)
    u_channel_budget = int(TOTAL_PARAM_BUDGET * U_BUDGET_PERCENT)
    v_channel_budget = int(TOTAL_PARAM_BUDGET * V_BUDGET_PERCENT)

    # When skipping HF training, assign ALL params to LL bands
    if SKIP_HF_TRAINING:
        y_hf_budget = 0
        u_hf_budget = 0
        v_hf_budget = 0
        y_budget = y_channel_budget  # Full channel budget for LL
        u_budget = u_channel_budget  # Full channel budget for LL
        v_budget = v_channel_budget  # Full channel budget for LL
    else:
        y_hf_budget = int(y_channel_budget * Y_HF_BUDGET_PERCENT)
        u_hf_budget = int(u_channel_budget * U_HF_BUDGET_PERCENT)
        v_hf_budget = int(v_channel_budget * V_HF_BUDGET_PERCENT)
        y_budget = y_channel_budget - y_hf_budget  # LL budget for Y
        u_budget = u_channel_budget - u_hf_budget  # LL budget for U
        v_budget = v_channel_budget - v_hf_budget  # LL budget for V
    
    # Get actual params for initial allocation
    y_ll_layers, y_ll_hidden = find_model_size_for_budget(y_budget, dim_in=2, dim_out=1, strict_under=True)
    y_ll_actual = calculate_model_params(y_ll_layers, y_ll_hidden, dim_in=2, dim_out=1)
    
    u_ll_layers, u_ll_hidden = find_model_size_for_budget(u_budget, dim_in=2, dim_out=1, strict_under=True)
    u_ll_actual = calculate_model_params(u_ll_layers, u_ll_hidden, dim_in=2, dim_out=1)
    
    v_ll_layers, v_ll_hidden = find_model_size_for_budget(v_budget, dim_in=2, dim_out=1, strict_under=True)
    v_ll_actual = calculate_model_params(v_ll_layers, v_ll_hidden, dim_in=2, dim_out=1)
    
    # Calculate leftover budget from LL bands
    total_ll_actual = y_ll_actual + u_ll_actual + v_ll_actual
    leftover = TOTAL_PARAM_BUDGET - total_ll_actual - (y_hf_budget + u_hf_budget + v_hf_budget)
    
    # Redistribute leftover to LL bands in priority: Y > U > V
    # But respect the TOTAL_PARAM_BUDGET constraint
    if leftover > 0 and SKIP_HF_TRAINING:
        # Try to add to Y first
        available_for_y = TOTAL_PARAM_BUDGET - (u_ll_actual + v_ll_actual)
        new_y_budget = min(y_budget + leftover, available_for_y)
        new_y_ll_layers, new_y_ll_hidden = find_model_size_for_budget(new_y_budget, dim_in=2, dim_out=1, strict_under=True)
        new_y_ll_actual = calculate_model_params(new_y_ll_layers, new_y_ll_hidden, dim_in=2, dim_out=1)
        
        if new_y_ll_actual > y_ll_actual and (new_y_ll_actual + u_ll_actual + v_ll_actual) <= TOTAL_PARAM_BUDGET:
            y_ll_layers = new_y_ll_layers
            y_ll_hidden = new_y_ll_hidden
            y_ll_actual = new_y_ll_actual
            leftover = TOTAL_PARAM_BUDGET - (y_ll_actual + u_ll_actual + v_ll_actual)
        
        # Try to add remaining to U
        if leftover > 0:
            available_for_u = TOTAL_PARAM_BUDGET - (y_ll_actual + v_ll_actual)
            new_u_budget = min(u_budget + leftover, available_for_u)
            new_u_ll_layers, new_u_ll_hidden = find_model_size_for_budget(new_u_budget, dim_in=2, dim_out=1, strict_under=True)
            new_u_ll_actual = calculate_model_params(new_u_ll_layers, new_u_ll_hidden, dim_in=2, dim_out=1)
            
            if new_u_ll_actual > u_ll_actual and (y_ll_actual + new_u_ll_actual + v_ll_actual) <= TOTAL_PARAM_BUDGET:
                u_ll_layers = new_u_ll_layers
                u_ll_hidden = new_u_ll_hidden
                u_ll_actual = new_u_ll_actual
                leftover = TOTAL_PARAM_BUDGET - (y_ll_actual + u_ll_actual + v_ll_actual)
        
        # Try to add remaining to V
        if leftover > 0:
            available_for_v = TOTAL_PARAM_BUDGET - (y_ll_actual + u_ll_actual)
            new_v_budget = min(v_budget + leftover, available_for_v)
            new_v_ll_layers, new_v_ll_hidden = find_model_size_for_budget(new_v_budget, dim_in=2, dim_out=1, strict_under=True)
            new_v_ll_actual = calculate_model_params(new_v_ll_layers, new_v_ll_hidden, dim_in=2, dim_out=1)
            
            if new_v_ll_actual > v_ll_actual and (y_ll_actual + u_ll_actual + new_v_ll_actual) <= TOTAL_PARAM_BUDGET:
                v_ll_layers = new_v_ll_layers
                v_ll_hidden = new_v_ll_hidden
                v_ll_actual = new_v_ll_actual
                leftover = TOTAL_PARAM_BUDGET - (y_ll_actual + u_ll_actual + v_ll_actual)
        
        # Update budgets to match actual allocations (for display purposes)
        y_budget = y_ll_actual
        u_budget = u_ll_actual
        v_budget = v_ll_actual
    
    total_ll_actual = y_ll_actual + u_ll_actual + v_ll_actual
    
    y_ll_percent = (y_budget / TOTAL_PARAM_BUDGET) * 100
    u_ll_percent = (u_budget / TOTAL_PARAM_BUDGET) * 100
    v_ll_percent = (v_budget / TOTAL_PARAM_BUDGET) * 100
    
    print(f"\nParameter Budget Allocation:")
    print(f"  Total Budget: {TOTAL_PARAM_BUDGET:,} parameters")
    if SKIP_HF_TRAINING:
        print(f"  Y Channel LL: {y_budget:,} ({y_ll_percent:.1f}%) → actual: {y_ll_actual:,}")
        print(f"  U Channel LL: {u_budget:,} ({u_ll_percent:.1f}%) → actual: {u_ll_actual:,}")
        print(f"  V Channel LL: {v_budget:,} ({v_ll_percent:.1f}%) → actual: {v_ll_actual:,}")
        print(f"  Total LL actual: {total_ll_actual:,} ({100*total_ll_actual/TOTAL_PARAM_BUDGET:.1f}% efficiency)")
    else:
        print(f"  Y Channel: {y_channel_budget:,} ({100*Y_BUDGET_PERCENT:.1f}%) = LL {y_budget:,} ({y_ll_percent:.1f}%) + HF {y_hf_budget:,} ({100*y_hf_budget/max(y_channel_budget, 1):.1f}%)")
        print(f"  U Channel: {u_channel_budget:,} ({100*U_BUDGET_PERCENT:.1f}%) = LL {u_budget:,} ({u_ll_percent:.1f}%) + HF {u_hf_budget:,} ({100*u_hf_budget/max(u_channel_budget, 1):.1f}%)")
        print(f"  V Channel: {v_channel_budget:,} ({100*V_BUDGET_PERCENT:.1f}%) = LL {v_budget:,} ({v_ll_percent:.1f}%) + HF {v_hf_budget:,} ({100*v_hf_budget/max(v_channel_budget, 1):.1f}%)")
    
    # Train models for each band individually with PSNR after each
    start_time = time.time()
    orig_h, orig_w = y_channel.shape
    
    # Initialize model dictionaries
    y_models = {}
    u_models = {}
    v_models = {}
    
    # Helper function to reconstruct and calc PSNR after each band
    band_psnrs = []  # Track PSNR after each band
    band_resource_usage = []  # Track per-band timing and memory usage
    def reconstruct_and_calc_psnr(band_desc):
        # Reconstruct channels
        y_rec_coeffs = reconstruct_channel_from_models(y_models, y_coeffs, device) if y_models else None
        u_rec_coeffs = reconstruct_channel_from_models(u_models, u_coeffs, device) if u_models else None
        v_rec_coeffs = reconstruct_channel_from_models(v_models, v_coeffs, device) if v_models else None
        
        if y_rec_coeffs is None or u_rec_coeffs is None or v_rec_coeffs is None:
            return
        
        # Inverse DWT
        y_rec = pywt.waverec2(y_rec_coeffs, WAVELET)[:orig_h, :orig_w]
        u_rec = pywt.waverec2(u_rec_coeffs, WAVELET)[:orig_h, :orig_w]
        v_rec = pywt.waverec2(v_rec_coeffs, WAVELET)[:orig_h, :orig_w]
        
        # Convert to RGB
        yuv_rec = np.stack([y_rec, u_rec, v_rec], axis=2)
        rgb_rec = yuv_to_rgb(yuv_rec)
        
        # Calculate PSNRs
        y_psnr = util.get_clamped_psnr(torch.FloatTensor(y_channel.flatten()), torch.FloatTensor(y_rec.flatten()))
        u_psnr = util.get_clamped_psnr(torch.FloatTensor(u_channel.flatten()), torch.FloatTensor(u_rec.flatten()))
        v_psnr = util.get_clamped_psnr(torch.FloatTensor(v_channel.flatten()), torch.FloatTensor(v_rec.flatten()))
        rgb_psnr = util.get_clamped_psnr(torch.FloatTensor(np.array(img_rgb).flatten()), torch.FloatTensor(rgb_rec.flatten()))
        
        print(f"  → After {band_desc}: Y={y_psnr:.2f}, U={u_psnr:.2f}, V={v_psnr:.2f}, RGB={rgb_psnr:.2f} dB")
        
        # Store PSNR
        band_psnrs.append({
            'band': band_desc,
            'y': float(y_psnr),
            'u': float(u_psnr),
            'v': float(v_psnr),
            'rgb': float(rgb_psnr)
        })
        
        return y_psnr, u_psnr, v_psnr, rgb_psnr
    
    # Define all 21 bands in training order
    # Format: (channel_name, channel, coeffs, budget, hf_budget, band_name)
    all_bands = [
        # LL bands (3 bands)
        ('Y', y_channel, y_coeffs, y_budget, y_hf_budget, 'LL'),
        ('U', u_channel, u_coeffs, u_budget, u_hf_budget, 'LL'),
        ('V', v_channel, v_coeffs, v_budget, v_hf_budget, 'LL'),
    ]
    
    # Level 1 HF bands (9 bands: 3 channels × 3 HF types)
    for band_type in ['cH_L1', 'cV_L1', 'cD_L1']:
        all_bands.append(('Y', y_channel, y_coeffs, y_budget, y_hf_budget, band_type))
        all_bands.append(('U', u_channel, u_coeffs, u_budget, u_hf_budget, band_type))
        all_bands.append(('V', v_channel, v_coeffs, v_budget, v_hf_budget, band_type))
    
    # Level 2 HF bands (9 bands: 3 channels × 3 HF types)
    for band_type in ['cH_L2', 'cV_L2', 'cD_L2']:
        all_bands.append(('Y', y_channel, y_coeffs, y_budget, y_hf_budget, band_type))
        all_bands.append(('U', u_channel, u_coeffs, u_budget, u_hf_budget, band_type))
        all_bands.append(('V', v_channel, v_coeffs, v_budget, v_hf_budget, band_type))
    
    print(f"\n{'='*70}")
    print(f"Training All 21 Sub-bands Individually")
    print(f"{'='*70}")
    
    # Train each band and reconstruct after each
    cuda_available = torch.cuda.is_available()
    for idx, (ch_name, ch_data, ch_coeffs, budget, hf_budget, band_name) in enumerate(all_bands, 1):
        band_desc = f"{ch_name}-{band_name}"
        print(f"\n[{idx}/21] Training {band_desc}...")

        # Track training time for this band
        band_start_time = time.time()
        
        # Train this specific band
        new_models = train_channel_dwt_models(
            ch_data, ch_coeffs, ch_name, budget, hf_budget, device, 
            train_only_band=band_name
        )
        
        if cuda_available:
            torch.cuda.synchronize()

        band_train_time = time.time() - band_start_time
        print(f"  Training time: {band_train_time:.2f} seconds")

        # Capture current memory utilization after training
        end_mem = get_memory_stats(device='cuda' if cuda_available else 'cpu')
        if end_mem is not None:
            current_allocated_mb = end_mem.get('current_allocated_mb', 0)
            peak_allocated_mb = end_mem.get('peak_allocated_mb', 0)
            current_reserved_mb = end_mem.get('current_reserved_mb', 0)
            peak_reserved_mb = end_mem.get('peak_reserved_mb', 0)
            print(
                f"  GPU Memory: {current_allocated_mb:.2f} MB allocated, "
                f"{peak_allocated_mb:.2f} MB peak allocated, "
                f"{current_reserved_mb:.2f} MB reserved, "
                f"{peak_reserved_mb:.2f} MB peak reserved"
            )
        else:
            current_allocated_mb = None
            peak_allocated_mb = None
            current_reserved_mb = None
            peak_reserved_mb = None
            print(f"  GPU Memory: (Unable to measure)")
        
        # Calculate and show parameter size
        if new_models:
            band_params = sum(info['params'] for info in new_models.values())
            band_params_kb = (band_params * 2) / 1024  # FP16 = 2 bytes per param
            print(f"  Parameters: {band_params:,} ({band_params_kb:.2f} KB)")

        band_resource_usage.append({
            'band': band_desc,
            'train_time_seconds': round(band_train_time, 4),
            'current_allocated_mb': round(current_allocated_mb, 2) if current_allocated_mb is not None else None,
            'peak_allocated_mb': round(peak_allocated_mb, 2) if peak_allocated_mb is not None else None,
            'current_reserved_mb': round(current_reserved_mb, 2) if current_reserved_mb is not None else None,
            'peak_reserved_mb': round(peak_reserved_mb, 2) if peak_reserved_mb is not None else None,
        })
        
        # Update the appropriate model dict
        if ch_name == 'Y':
            y_models.update(new_models)
        elif ch_name == 'U':
            u_models.update(new_models)
        else:  # V
            v_models.update(new_models)
        
        # Reconstruct and show PSNR
        reconstruct_and_calc_psnr(band_desc)
    
    train_time = time.time() - start_time
    
    # Calculate actual total parameters
    actual_params_y = sum(info['params'] for info in y_models.values())
    actual_params_u = sum(info['params'] for info in u_models.values())
    actual_params_v = sum(info['params'] for info in v_models.values())
    total_actual_params = actual_params_y + actual_params_u + actual_params_v
    
    # Reconstruct final image and save it
    print(f"\nReconstructing final image...")
    y_rec_coeffs = reconstruct_channel_from_models(y_models, y_coeffs, device)
    u_rec_coeffs = reconstruct_channel_from_models(u_models, u_coeffs, device)
    v_rec_coeffs = reconstruct_channel_from_models(v_models, v_coeffs, device)
    
    y_rec = pywt.waverec2(y_rec_coeffs, WAVELET)[:orig_h, :orig_w]
    u_rec = pywt.waverec2(u_rec_coeffs, WAVELET)[:orig_h, :orig_w]
    v_rec = pywt.waverec2(v_rec_coeffs, WAVELET)[:orig_h, :orig_w]
    
    yuv_rec = np.stack([y_rec, u_rec, v_rec], axis=2)
    rgb_rec = yuv_to_rgb(yuv_rec)
    
    # Save reconstructed image
    reconstructed_img_path = os.path.join(LOG_DIR, f"{IMAGEID}_reconstructed.png")
    Image.fromarray(rgb_rec).save(reconstructed_img_path)
    print(f"Saved reconstructed image to: {reconstructed_img_path}")
    
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
    print("\nSaving models (converted to FP16 for storage)...")
    models_dir = os.path.join(LOG_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    total_model_size_bytes = 0
    
    # Save Y channel models
    for band_name, info in y_models.items():
        model_path = os.path.join(models_dir, f"y_{band_name.lower()}_model.pt")
        # Convert to FP16 for storage (like in main.py)
        model_fp16 = {k: v.half() if v.dtype in [torch.float32, torch.float64] else v 
                      for k, v in info['model'].state_dict().items()}
        torch.save(model_fp16, model_path)
        model_size = os.path.getsize(model_path)
        model_size_kb = model_size / 1024
        params = info['params']
        pure_param_size_bytes = params * 2  # FP16 = 2 bytes per param
        pure_param_size_kb = pure_param_size_bytes / 1024
        overhead_bytes = model_size - pure_param_size_bytes
        overhead_pct = (overhead_bytes / model_size * 100) if model_size > 0 else 0
        total_model_size_bytes += model_size
        print(f"  Y {band_name}: {model_size_kb:.2f} KB (params: {pure_param_size_kb:.2f} KB, overhead: {overhead_bytes/1024:.2f} KB, {overhead_pct:.1f}%)")
    
    # Save U channel models
    for band_name, info in u_models.items():
        model_path = os.path.join(models_dir, f"u_{band_name.lower()}_model.pt")
        # Convert to FP16 for storage (like in main.py)
        model_fp16 = {k: v.half() if v.dtype in [torch.float32, torch.float64] else v 
                      for k, v in info['model'].state_dict().items()}
        torch.save(model_fp16, model_path)
        model_size = os.path.getsize(model_path)
        model_size_kb = model_size / 1024
        params = info['params']
        pure_param_size_bytes = params * 2  # FP16 = 2 bytes per param
        pure_param_size_kb = pure_param_size_bytes / 1024
        overhead_bytes = model_size - pure_param_size_bytes
        overhead_pct = (overhead_bytes / model_size * 100) if model_size > 0 else 0
        total_model_size_bytes += model_size
        print(f"  U {band_name}: {model_size_kb:.2f} KB (params: {pure_param_size_kb:.2f} KB, overhead: {overhead_bytes/1024:.2f} KB, {overhead_pct:.1f}%)")
    
    # Save V channel models
    for band_name, info in v_models.items():
        model_path = os.path.join(models_dir, f"v_{band_name.lower()}_model.pt")
        # Convert to FP16 for storage (like in main.py)
        model_fp16 = {k: v.half() if v.dtype in [torch.float32, torch.float64] else v 
                      for k, v in info['model'].state_dict().items()}
        torch.save(model_fp16, model_path)
        model_size = os.path.getsize(model_path)
        model_size_kb = model_size / 1024
        params = info['params']
        pure_param_size_bytes = params * 2  # FP16 = 2 bytes per param
        pure_param_size_kb = pure_param_size_bytes / 1024
        overhead_bytes = model_size - pure_param_size_bytes
        overhead_pct = (overhead_bytes / model_size * 100) if model_size > 0 else 0
        total_model_size_bytes += model_size
        print(f"  V {band_name}: {model_size_kb:.2f} KB (params: {pure_param_size_kb:.2f} KB, overhead: {overhead_bytes/1024:.2f} KB, {overhead_pct:.1f}%)")
    
    total_model_size_kb = total_model_size_bytes / 1024
    total_model_size_mb = total_model_size_kb / 1024
    total_model_param_bytes = total_actual_params * 2  # FP16 storage per parameter
    total_model_bpp = (total_model_param_bytes * 8) / (orig_h * orig_w * 3)
    
    print(f"\n  Total Model Size: {total_model_size_bytes:,} bytes ({total_model_size_kb:.2f} KB, {total_model_size_mb:.2f} MB)")
    print(f"  Total Param Size: {total_model_param_bytes:,} bytes ({total_model_param_bytes/1024:.2f} KB)")
    print(f"  Total Model BPP: {total_model_bpp:.4f} bits/pixel")
    print(f"  Saved to: {models_dir}")
    
    # Final summary showing PSNR progression
    print(f"\n{'='*70}")
    print(f"Training Complete! PSNR Progression Summary:")
    print(f"{'='*70}")
    if band_psnrs:
        final_psnr = band_psnrs[-1]
        print(f"Final PSNR: Y={final_psnr['y']:.2f}, U={final_psnr['u']:.2f}, V={final_psnr['v']:.2f}, RGB={final_psnr['rgb']:.2f} dB")
        print(f"Total training time: {train_time:.2f} seconds")
        print(f"See above for per-band PSNR progression")
    print(f"{'='*70}")
    
    # Save results with per-band PSNR progression
    final_psnr = band_psnrs[-1] if band_psnrs else {'y': 0, 'u': 0, 'v': 0, 'rgb': 0}
    results = {
        'image_id': IMAGEID,
        'reconstructed_image': reconstructed_img_path,
        'total_params': total_actual_params,
        'params_y': actual_params_y,
        'params_u': actual_params_u,
        'params_v': actual_params_v,
        'param_budget': TOTAL_PARAM_BUDGET,
        'y_psnr': final_psnr['y'],
        'u_psnr': final_psnr['u'],
        'v_psnr': final_psnr['v'],
        'rgb_psnr': final_psnr['rgb'],
        'band_psnrs': band_psnrs,  # PSNR after each of 21 bands
        'band_resource_usage': band_resource_usage,
        'train_time': train_time,
        'levels': LEVELS,
        'wavelet': WAVELET,
        'total_model_size_bytes': total_model_size_bytes,
        'total_model_size_kb': round(total_model_size_kb, 2),
        'total_model_size_mb': round(total_model_size_mb, 2),
        'total_model_param_size_bytes': total_model_param_bytes,
        'total_model_param_size_kb': round(total_model_param_bytes / 1024, 2),
        'total_model_bpp': round(total_model_bpp, 6),
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
    
    # Log final resource statistics
    print_memory_stats("Memory at script end", device='cuda' if torch.cuda.is_available() else 'cpu')
    monitor.print_summary()
    
    # Save resource summary
    resource_summary_path = os.path.join(LOG_DIR, f"{IMAGEID}_resource_summary.json")
    monitor.save_summary(resource_summary_path)

if __name__ == "__main__":
    main()
