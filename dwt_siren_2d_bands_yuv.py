import os
import random
import json
import time
import numpy as np
import pywt
import torch
from PIL import Image
from torchvision import transforms
from training import Trainer
from siren import Siren
import util

# ---------------------------
# CONFIG
LEVELS = 2        # Number of DWT decomposition levels
WAVELET = "db4"   # Wavelet type: 'db1','db4','sym4', etc.
LOG_DIR = "results/dwt_2d_bands_yuv"

# Chroma subsampling (4:2:0 standard)
USE_CHROMA_SUBSAMPLING = True  # True: 4:2:0 subsampling for UV channels
CHROMA_SUBSAMPLE_FACTOR = 2    # Downsample UV by this factor in each dimension

# # Model Configuration (Maximum/Fallback - Adaptive sizing based on coefficient count)
# MODEL_NUM_LAYERS = 40   # Max layers for very large inputs
# MODEL_LAYER_SIZE = 128  # Max hidden units for large inputs
# MODEL_ITERATIONS = 2000 # Max training iterations

# # Model width scaling - different for Y vs UV channels
# MODEL_SIZE_SCALE_Y_LL = 0.6  # Y LL model scale
# MODEL_SIZE_SCALE_Y_HF = 0.5  # Y HF model scale
# MODEL_SIZE_SCALE_UV_LL = 0.25  # UV LL model scale (aggressive - ~6% params)
# MODEL_SIZE_SCALE_UV_HF = 0.15  # UV HF model scale (aggressive - ~2% params)


# Model Configuration (Maximum/Fallback - Adaptive sizing based on coefficient count)
MODEL_NUM_LAYERS = 20   # Max layers (reduced from 40 for smaller model)
MODEL_LAYER_SIZE = 64   # Max hidden units (reduced from 128 for smaller model)
MODEL_ITERATIONS = 1000 # Max training iterations (reduced from 2000 for faster training)

# Model width scaling - different for Y vs UV channels
MODEL_SIZE_SCALE_Y_LL = 0.4   # Y LL model scale (reduced from 0.6)
MODEL_SIZE_SCALE_Y_HF = 0.3   # Y HF model scale (reduced from 0.5)
MODEL_SIZE_SCALE_UV_LL = 0.2  # UV LL model scale (reduced from 0.25)
MODEL_SIZE_SCALE_UV_HF = 0.1  # UV HF model scale (reduced from 0.15)


# High-frequency coefficient handling
USE_SPARSE_HF = False  # True: Use sparse threshold training for HF
THRESHOLD_FACTOR = 1  # Keep coefficients above this * std
LEVEL_THRESHOLD_GAMMA = 0.8  # Per-level threshold scaling

IMAGEID = "kodim02"

OUTPUT_FILE = f"{IMAGEID}_dwt_yuv_levels{LEVELS}_{WAVELET}_thresh{THRESHOLD_FACTOR}_gamma{LEVEL_THRESHOLD_GAMMA}.png"

# ---------------------------

def extract_2d_sparse_coeffs(band, threshold):
    """Extract sparse coefficients with their 2D coordinates from a band"""
    significant_mask = np.abs(band) > threshold
    rows, cols = np.where(significant_mask)
    values = band[rows, cols]
    
    h, w = band.shape
    coords_y = (rows / (h - 1)) * 2 - 1
    coords_x = (cols / (w - 1)) * 2 - 1
    coords = np.stack([coords_x, coords_y], axis=1)
    
    return values, coords, rows, cols, significant_mask.sum()

def process_channel_dwt(channel_data, channel_name, ll_scale, hf_scale, is_chroma=False):
    """Process DWT for a single channel and return models and data"""
    import math
    
    # Perform DWT
    coeffs = pywt.wavedec2(channel_data, wavelet=WAVELET, level=LEVELS)
    ll_coeffs = coeffs[0]
    
    print(f"\n=== {channel_name} Channel ===")
    print(f"LL coefficients shape: {ll_coeffs.shape}")
    
    # Storage for band data
    band_data_per_level = []
    
    if USE_SPARSE_HF:
        # Collect all HF coefficients
        all_hf_coeffs = []
        for level_idx in range(1, len(coeffs)):
            cH, cV, cD = coeffs[level_idx]
            all_hf_coeffs.extend(cH.flatten())
            all_hf_coeffs.extend(cV.flatten())
            all_hf_coeffs.extend(cD.flatten())
        all_hf_coeffs = np.array(all_hf_coeffs)

        hf_std = np.std(all_hf_coeffs)
        base_threshold = THRESHOLD_FACTOR * hf_std
        
        print(f"HF std: {hf_std:.2f}, Base threshold: {base_threshold:.2f}")

        total_coeffs = 0
        total_significant = 0

        # Extract sparse coefficients
        for level_idx in range(1, len(coeffs)):
            cH, cV, cD = coeffs[level_idx]
            level_threshold = base_threshold * (LEVEL_THRESHOLD_GAMMA ** (level_idx - 1))

            cH_vals, cH_coords, cH_rows, cH_cols, cH_count = extract_2d_sparse_coeffs(cH, level_threshold)
            cV_vals, cV_coords, cV_rows, cV_cols, cV_count = extract_2d_sparse_coeffs(cV, level_threshold)
            cD_vals, cD_coords, cD_rows, cD_cols, cD_count = extract_2d_sparse_coeffs(cD, level_threshold)

            level_total = cH.size + cV.size + cD.size
            level_significant = cH_count + cV_count + cD_count
            total_coeffs += level_total
            total_significant += level_significant

            print(f"Level {level_idx}: Significant {level_significant}/{level_total} ({100*level_significant/level_total:.1f}%)")

            band_data_per_level.append({
                'cH': {'values': cH_vals, 'coords': cH_coords, 'rows': cH_rows, 'cols': cH_cols, 'shape': cH.shape},
                'cV': {'values': cV_vals, 'coords': cV_coords, 'rows': cV_rows, 'cols': cV_cols, 'shape': cV.shape},
                'cD': {'values': cD_vals, 'coords': cD_coords, 'rows': cD_rows, 'cols': cD_cols, 'shape': cD.shape},
                'threshold': level_threshold
            })

        print(f"Overall sparsity: {100*(1-total_significant/total_coeffs):.2f}%")
    else:
        for level_idx in range(1, len(coeffs)):
            cH, cV, cD = coeffs[level_idx]
            band_data_per_level.append({
                'cH': {'values': np.array([]), 'coords': np.array([]), 'shape': cH.shape},
                'cV': {'values': np.array([]), 'coords': np.array([]), 'shape': cV.shape},
                'cD': {'values': np.array([]), 'coords': np.array([]), 'shape': cD.shape}
            })

    # Setup device
    dtype = torch.float32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Normalize LL coefficients
    ll_mean = ll_coeffs.mean()
    ll_std = ll_coeffs.std()
    ll_norm = (ll_coeffs - ll_mean) / (ll_std + 1e-8)
    
    coeffs_tensor = transforms.ToTensor()(ll_norm).float().to(device, dtype)

    # Adaptive LL model sizing
    ll_h, ll_w = ll_coeffs.shape
    ll_pixels = ll_h * ll_w
    log_pixels = math.log10(max(ll_pixels, 10))
    
    adaptive_ll_layers = max(3, min(MODEL_NUM_LAYERS, int(3.5 * log_pixels - 4)))
    base_ll_size = max(32, min(MODEL_LAYER_SIZE, int(32 * log_pixels - 32)))
    adaptive_ll_size = max(16, int(base_ll_size * ll_scale))
    adaptive_ll_iters = 2500 if ll_pixels < 1000 else (2000 if ll_pixels < 3000 else MODEL_ITERATIONS)
    
    # Reduce iterations for chroma channels (U/V)
    if is_chroma:
        adaptive_ll_iters = min(1000, adaptive_ll_iters // 3)
    
    print(f"LL model: {adaptive_ll_layers}L × {adaptive_ll_size}U ({adaptive_ll_iters} iters)")

    # Create LL model
    func_rep_ll = Siren(
        dim_in=2,
        dim_hidden=adaptive_ll_size,
        dim_out=1,
        num_layers=adaptive_ll_layers,
        final_activation=torch.nn.Identity(),
        w0_initial=30.0,
        w0=30.0
    )
    
    # Helper function for adaptive HF parameters
    def get_adaptive_hf_params(num_coeffs):
        log_coeffs = math.log10(max(num_coeffs, 10))
        layers = max(3, min(MODEL_NUM_LAYERS, int(3.5 * log_coeffs - 4)))
        base_size = max(32, min(MODEL_LAYER_SIZE, int(32 * log_coeffs - 32)))
        size = max(16, int(base_size * hf_scale))
        iters = 2500 if num_coeffs < 1000 else (2000 if num_coeffs < 5000 else (1500 if num_coeffs < 20000 else MODEL_ITERATIONS))
        
        # Reduce iterations for chroma channels (U/V)
        if is_chroma:
            iters = min(800, iters // 3)
        
        return layers, size, iters
    
    # Initialize HF models
    band_models = []
    
    if USE_SPARSE_HF:
        for level_idx, level_data in enumerate(band_data_per_level):
            level_models = {}
            
            for band_name in ['cH', 'cV', 'cD']:
                if len(level_data[band_name]['values']) > 0:
                    num_coeffs = len(level_data[band_name]['values'])
                    hf_layers, hf_size, hf_iters = get_adaptive_hf_params(num_coeffs)
                    
                    model = Siren(
                        dim_in=2,
                        dim_hidden=hf_size,
                        dim_out=1,
                        num_layers=hf_layers,
                        final_activation=torch.nn.Identity(),
                        w0_initial=30.0,
                        w0=30.0
                    )
                    level_models[band_name] = {
                        'model': model,
                        'values': level_data[band_name]['values'],
                        'coords': level_data[band_name]['coords'],
                        'rows': level_data[band_name]['rows'],
                        'cols': level_data[band_name]['cols'],
                        'shape': level_data[band_name]['shape'],
                        'adaptive_layers': hf_layers,
                        'adaptive_size': hf_size,
                        'adaptive_iters': hf_iters
                    }
            
            band_models.append(level_models)

    return {
        'll_coeffs': ll_coeffs,
        'll_mean': ll_mean,
        'll_std': ll_std,
        'll_norm': ll_norm,
        'coeffs_tensor': coeffs_tensor,
        'func_rep_ll': func_rep_ll,
        'adaptive_ll_layers': adaptive_ll_layers,
        'adaptive_ll_size': adaptive_ll_size,
        'adaptive_ll_iters': adaptive_ll_iters,
        'band_models': band_models,
        'band_data_per_level': band_data_per_level
    }

def train_channel_models(channel_data, channel_name):
    """Train all models for a channel"""
    dtype = torch.float32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train LL model
    print(f"\n=== Training {channel_name} LL Model ===")
    trainer_ll = Trainer(channel_data['func_rep_ll'], lr=2e-4)
    coordinates_ll, features_ll = util.to_coordinates_and_coeffs_features(channel_data['coeffs_tensor'])
    coordinates_ll, features_ll = coordinates_ll.to(device, dtype), features_ll.to(device, dtype)
    
    trainer_ll.train(coordinates_ll, features_ll, num_iters=channel_data['adaptive_ll_iters'])
    print(f'Best {channel_name} LL PSNR: {trainer_ll.best_vals["psnr"]:.2f}')
    
    channel_data['trainer_ll'] = trainer_ll
    channel_data['coordinates_ll'] = coordinates_ll
    
    # Train HF models
    if USE_SPARSE_HF and channel_data['band_models']:
        print(f"\n=== Training {channel_name} HF Models ===")
        for level_idx, level_models in enumerate(channel_data['band_models']):
            for band_name in ['cH', 'cV', 'cD']:
                if band_name in level_models:
                    band_info = level_models[band_name]
                    num_coeffs = len(band_info['values'])
                    
                    print(f"  Level {level_idx+1} {band_name}: {band_info['adaptive_layers']}L×{band_info['adaptive_size']}U ({num_coeffs} coeffs)")
                    
                    # Normalize
                    values = band_info['values']
                    val_mean = values.mean()
                    val_std = values.std()
                    val_norm = (values - val_mean) / (val_std + 1e-8)
                    
                    band_info['mean'] = val_mean
                    band_info['std'] = val_std
                    
                    # Train
                    coords_tensor = torch.tensor(band_info['coords'], dtype=torch.float32).to(device, dtype)
                    features_tensor = torch.tensor(val_norm.reshape(-1, 1), dtype=torch.float32).to(device, dtype)
                    
                    trainer = Trainer(band_info['model'], lr=2e-4)
                    trainer.train(coords_tensor, features_tensor, num_iters=band_info['adaptive_iters'])
                    
                    print(f"    → PSNR={trainer.best_vals['psnr']:.2f}")
                    band_info['trainer'] = trainer

def reconstruct_channel(channel_data, channel_name):
    """Reconstruct a channel from trained models"""
    dtype = torch.float32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load best LL model
    channel_data['func_rep_ll'].load_state_dict(channel_data['trainer_ll'].best_model)
    channel_data['func_rep_ll'] = channel_data['func_rep_ll'].half().to('cuda')
    coordinates_ll = channel_data['coordinates_ll'].half().to('cuda')
    
    # Load best HF models
    if USE_SPARSE_HF:
        for level_models in channel_data['band_models']:
            for band_name, band_info in level_models.items():
                band_info['model'].load_state_dict(band_info['trainer'].best_model)
                band_info['model'] = band_info['model'].half().to('cuda')

    with torch.no_grad():
        # Reconstruct LL
        ll_recon = channel_data['func_rep_ll'](coordinates_ll).reshape(
            channel_data['coeffs_tensor'].shape[1], 
            channel_data['coeffs_tensor'].shape[2]
        ).float()
        ll_recon_denorm = ll_recon * (channel_data['ll_std'] + 1e-8) + channel_data['ll_mean']
        ll_recon_np = ll_recon_denorm.cpu().numpy()
        
        # Build coefficient structure
        coeffs_modified = [ll_recon_np]
        
        # Reconstruct HF bands
        if USE_SPARSE_HF:
            for level_idx, level_models in enumerate(channel_data['band_models']):
                cH_recon = np.zeros(channel_data['band_data_per_level'][level_idx]['cH']['shape'])
                cV_recon = np.zeros(channel_data['band_data_per_level'][level_idx]['cV']['shape'])
                cD_recon = np.zeros(channel_data['band_data_per_level'][level_idx]['cD']['shape'])
                
                for band_name, band_array in [('cH', cH_recon), ('cV', cV_recon), ('cD', cD_recon)]:
                    if band_name in level_models:
                        band_info = level_models[band_name]
                        
                        trained_coords_tensor = torch.tensor(band_info['coords'], dtype=torch.float32).to(device).half()
                        predictions = band_info['model'](trained_coords_tensor).reshape(-1).float()
                        predictions_denorm = predictions * (band_info['std'] + 1e-8) + band_info['mean']
                        predictions_np = predictions_denorm.cpu().numpy()
                        
                        band_array[band_info['rows'], band_info['cols']] = predictions_np
                
                coeffs_modified.append((cH_recon, cV_recon, cD_recon))
        else:
            for level_data in channel_data['band_data_per_level']:
                cH_zero = np.zeros(level_data['cH']['shape'])
                cV_zero = np.zeros(level_data['cV']['shape'])
                cD_zero = np.zeros(level_data['cD']['shape'])
                coeffs_modified.append((cH_zero, cV_zero, cD_zero))
        
        # Inverse DWT
        channel_recon = pywt.waverec2(coeffs_modified, wavelet=WAVELET)
        channel_recon = np.clip(channel_recon, 0, 255)
        
        return channel_recon

def calculate_model_size(channel_data, channel_name):
    """Calculate total model size for a channel"""
    model_size_ll = util.model_size_in_bits(channel_data['func_rep_ll']) / 8000.
    print(f'\n{channel_name} LL Model: {model_size_ll:.1f}kB')
    
    total_size = model_size_ll
    
    if USE_SPARSE_HF and channel_data['band_models']:
        total_hf = 0
        for level_idx, level_models in enumerate(channel_data['band_models']):
            for band_name, band_info in level_models.items():
                band_size = util.model_size_in_bits(band_info['model']) / 8000.
                total_hf += band_size
        print(f'{channel_name} HF Models: {total_hf:.1f}kB')
        total_size += total_hf
    
    print(f'{channel_name} Total: {total_size:.1f}kB')
    return total_size

def main():
    start_time = time.time()
    
    # Load image and convert to YCbCr
    img_rgb = Image.open(f"kodak-dataset/{IMAGEID}.png")
    img_ycbcr = img_rgb.convert("YCbCr")
    y_channel, u_channel, v_channel = img_ycbcr.split()
    
    # Convert to numpy arrays
    Y = np.asarray(y_channel, dtype=np.float32)
    U_full = np.asarray(u_channel, dtype=np.float32)
    V_full = np.asarray(v_channel, dtype=np.float32)
    
    # Apply 4:2:0 chroma subsampling if enabled
    if USE_CHROMA_SUBSAMPLING:
        # Downsample U and V channels
        u_img = Image.fromarray(U_full.astype(np.uint8))
        v_img = Image.fromarray(V_full.astype(np.uint8))
        
        new_size = (u_img.width // CHROMA_SUBSAMPLE_FACTOR, u_img.height // CHROMA_SUBSAMPLE_FACTOR)
        u_downsampled = u_img.resize(new_size, Image.Resampling.LANCZOS)
        v_downsampled = v_img.resize(new_size, Image.Resampling.LANCZOS)
        
        U = np.asarray(u_downsampled, dtype=np.float32)
        V = np.asarray(v_downsampled, dtype=np.float32)
        
        print(f"Original image shape: {Y.shape}")
        print(f"4:2:0 Chroma subsampling applied:")
        print(f"  Y channel: {Y.shape}")
        print(f"  U channel: {U.shape} (downsampled from {U_full.shape})")
        print(f"  V channel: {V.shape} (downsampled from {V_full.shape})")
    else:
        U = U_full
        V = V_full
        print(f"Original image shape: {Y.shape}")
        print(f"No chroma subsampling (4:4:4)")
    
    original_size_bytes = Y.nbytes + U.nbytes + V.nbytes
    original_size_kb = original_size_bytes / 1024
    original_bpp = (original_size_bytes * 8) / (Y.shape[0] * Y.shape[1])
    print(f"Original size: {original_size_kb:.2f} kB ({original_bpp:.2f} bpp)")
    
    # Set random seed
    torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')
    seed = random.randint(1, int(1e6))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Create output directory
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Process each channel
    y_data = process_channel_dwt(Y, "Y", MODEL_SIZE_SCALE_Y_LL, MODEL_SIZE_SCALE_Y_HF, is_chroma=False)
    u_data = process_channel_dwt(U, "U", MODEL_SIZE_SCALE_UV_LL, MODEL_SIZE_SCALE_UV_HF, is_chroma=True)
    v_data = process_channel_dwt(V, "V", MODEL_SIZE_SCALE_UV_LL, MODEL_SIZE_SCALE_UV_HF, is_chroma=True)
    
    # Train all models
    train_channel_models(y_data, "Y")
    train_channel_models(u_data, "U")
    train_channel_models(v_data, "V")
    
    # Calculate total model size
    print("\n" + "="*60)
    print("MODEL SIZES")
    print("="*60)
    y_size = calculate_model_size(y_data, "Y")
    u_size = calculate_model_size(u_data, "U")
    v_size = calculate_model_size(v_data, "V")
    
    total_model_size = y_size + u_size + v_size
    print(f"\nTotal Model Size (FP32): {total_model_size:.1f}kB")
    print(f"Total Model Size (FP16): {total_model_size/2:.1f}kB")
    
    # Reconstruct channels
    print("\n" + "="*60)
    print("RECONSTRUCTION")
    print("="*60)
    
    Y_recon = reconstruct_channel(y_data, "Y")
    U_recon = reconstruct_channel(u_data, "U")
    V_recon = reconstruct_channel(v_data, "V")
    
    # Upsample U and V if 4:2:0 was used
    if USE_CHROMA_SUBSAMPLING:
        # Upsample U and V back to Y resolution
        target_size = (Y_recon.shape[1], Y_recon.shape[0])  # (width, height)
        
        u_img_small = Image.fromarray(U_recon.astype(np.uint8))
        v_img_small = Image.fromarray(V_recon.astype(np.uint8))
        
        u_img_upsampled = u_img_small.resize(target_size, Image.Resampling.LANCZOS)
        v_img_upsampled = v_img_small.resize(target_size, Image.Resampling.LANCZOS)
        
        U_recon_full = np.asarray(u_img_upsampled, dtype=np.uint8)
        V_recon_full = np.asarray(v_img_upsampled, dtype=np.uint8)
        
        print(f"Upsampled U/V from {U_recon.shape} to {U_recon_full.shape}")
    else:
        # Ensure same dimensions (no subsampling case)
        min_h = min(Y_recon.shape[0], U_recon.shape[0], V_recon.shape[0])
        min_w = min(Y_recon.shape[1], U_recon.shape[1], V_recon.shape[1])
        
        U_recon_full = U_recon[:min_h, :min_w].astype(np.uint8)
        V_recon_full = V_recon[:min_h, :min_w].astype(np.uint8)
    
    Y_recon = Y_recon.astype(np.uint8)
    if USE_CHROMA_SUBSAMPLING:
        # Ensure Y matches upsampled UV size
        min_h = min(Y_recon.shape[0], U_recon_full.shape[0])
        min_w = min(Y_recon.shape[1], U_recon_full.shape[1])
        Y_recon = Y_recon[:min_h, :min_w]
        U_recon_full = U_recon_full[:min_h, :min_w]
        V_recon_full = V_recon_full[:min_h, :min_w]
    
    # Merge YUV and convert to RGB
    ycbcr_recon = Image.merge("YCbCr", [
        Image.fromarray(Y_recon),
        Image.fromarray(U_recon_full),
        Image.fromarray(V_recon_full)
    ])
    rgb_recon = ycbcr_recon.convert("RGB")
    rgb_recon.save(OUTPUT_FILE)
    
    # Calculate PSNR
    original_rgb = np.asarray(img_rgb, dtype=np.float32)
    recon_rgb = np.asarray(rgb_recon, dtype=np.float32)
    img_psnr = util.calc_psnr(original_rgb, recon_rgb)
    
    # Calculate per-channel PSNR
    y_psnr = util.calc_psnr(Y, Y_recon)
    
    if USE_CHROMA_SUBSAMPLING:
        # For subsampled channels, compare at subsampled resolution
        u_psnr = util.calc_psnr(U, U_recon.astype(np.float32))
        v_psnr = util.calc_psnr(V, V_recon.astype(np.float32))
    else:
        u_psnr = util.calc_psnr(U, U_recon_full.astype(np.float32))
        v_psnr = util.calc_psnr(V, V_recon_full.astype(np.float32))
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    if USE_CHROMA_SUBSAMPLING:
        print(f"Chroma subsampling:  4:2:0 (UV at 1/{CHROMA_SUBSAMPLE_FACTOR}x resolution)")
    else:
        print(f"Chroma subsampling:  None (4:4:4)")
    print(f"Original size:       {original_size_kb:.2f} kB ({original_bpp:.2f} bpp)")
    print(f"Model size (FP16):   {total_model_size/2:.1f} kB")
    print(f"Compression ratio:   {original_size_kb/(total_model_size/2):.2f}x")
    print(f"\nRGB Image PSNR:      {img_psnr:.2f} dB")
    print(f"Y Channel PSNR:      {y_psnr:.2f} dB")
    print(f"U Channel PSNR:      {u_psnr:.2f} dB")
    print(f"V Channel PSNR:      {v_psnr:.2f} dB")
    print(f"\nTotal time:          {total_time:.2f}s ({total_time/60:.2f} min)")
    print("="*60)
    
    print(f"\nSaved reconstructed image to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
