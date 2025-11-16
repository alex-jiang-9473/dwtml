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
LEVELS = 1        # Number of DWT decomposition levels
WAVELET = "db4"   # Wavelet type: 'db1','db4','sym4', etc.
LOG_DIR = "results/dwt_2d_bands"

# Model Configuration (Maximum/Fallback - Adaptive sizing based on coefficient count)
# LL model auto-scales based on pixel count: 3-40 layers, 32-128 units, 1000-2000 iters
# HF models auto-scale per band based on coeffs: 3-20 layers, 32-128 units, 1000-2000 iters
MODEL_NUM_LAYERS = 40   # Max layers for very large inputs (LL>=5000px or HF>30k coeffs)
MODEL_LAYER_SIZE = 128  # Max hidden units for large inputs. Auto-scales down for smaller inputs
MODEL_ITERATIONS = 2000 # Max training iterations. Auto-scales up for small/complex inputs

# Model width scaling (hidden units). Since params ~ width^2:
MODEL_SIZE_SCALE_LL = 0.6  # LL model scale: 0.6 → ~36% params (smaller for compression)
MODEL_SIZE_SCALE_HF = 0.5  # HF model scale: 0.5 → ~25% params (smaller for compression)

# High-frequency coefficient handling mode
USE_SPARSE_HF = True  # True: Use sparse threshold training for HF, False: Fill HF with zeros
THRESHOLD_FACTOR = 1  # Only used when USE_SPARSE_HF=True. Keep coefficients above this * std
# Range: 0.01 - 2.00 for sparse, 100+ to zero out all HF (black). Lower values keep more coefficients

# Per-level threshold scaling: effective_threshold(level) = base_threshold * (LEVEL_THRESHOLD_GAMMA ** (level_idx-1))
# level_idx starts from 1 for the finest level (highest resolution details). Set <1 to keep more at coarser levels.
LEVEL_THRESHOLD_GAMMA = 0.8

IMAGEID = "kodim02"

OUTPUT_FILE = f"{IMAGEID}_dwt_2d_bands_levels{LEVELS}_{WAVELET}_thresh{THRESHOLD_FACTOR}_gamma{LEVEL_THRESHOLD_GAMMA}.png"

# ---------------------------

def extract_2d_sparse_coeffs(band, threshold):
    """Extract sparse coefficients with their 2D coordinates from a band"""
    # Find significant coefficients (above threshold)
    significant_mask = np.abs(band) > threshold
    
    # Get 2D indices where coefficients are significant
    rows, cols = np.where(significant_mask)
    
    # Get the values at those positions
    values = band[rows, cols]
    
    # Create normalized 2D coordinates in [-1, 1]
    h, w = band.shape
    coords_y = (rows / (h - 1)) * 2 - 1  # Normalize to [-1, 1]
    coords_x = (cols / (w - 1)) * 2 - 1  # Normalize to [-1, 1]
    coords = np.stack([coords_x, coords_y], axis=1)  # Shape: (N, 2)
    
    # Also return the indices for reconstruction
    return values, coords, rows, cols, significant_mask.sum()

def main():
    # Start timing
    start_time = time.time()
    
    # 1) Load and prepare grayscale image
    img = Image.open(f"kodak-dataset/{IMAGEID}.png").convert("L")
    A = np.asarray(img, dtype=np.float32)
    
    # Calculate original image size
    original_size_bytes = A.nbytes
    original_size_kb = original_size_bytes / 1024
    original_bpp = (original_size_bytes * 8) / (A.shape[0] * A.shape[1])

    # 2) Perform multi-level 2D DWT
    coeffs = pywt.wavedec2(A, wavelet=WAVELET, level=LEVELS)

    # 3) Extract LL (approximation) coefficients
    ll_coeffs = coeffs[0]
    
    print(f"Original image shape: {A.shape}")
    print(f"Original image size: {original_size_kb:.2f} kB ({original_bpp:.2f} bits per pixel)")
    print(f"LL coefficients shape: {ll_coeffs.shape}")
    print(f"High-frequency mode: {'Sparse 2D coordinates' if USE_SPARSE_HF else 'Zero-filled (LL only)'}")
    
    # Storage for each band's data across all levels
    # Format: List of (cH_data, cV_data, cD_data) per level
    band_data_per_level = []
    
    if USE_SPARSE_HF:
        # Collect all HF coefficients to compute threshold (global stats)
        all_hf_coeffs = []
        for level_idx in range(1, len(coeffs)):
            cH, cV, cD = coeffs[level_idx]
            all_hf_coeffs.extend(cH.flatten())
            all_hf_coeffs.extend(cV.flatten())
            all_hf_coeffs.extend(cD.flatten())
        all_hf_coeffs = np.array(all_hf_coeffs)

        hf_std = np.std(all_hf_coeffs)
        base_threshold = THRESHOLD_FACTOR * hf_std

        print(f"\nHigh-frequency std: {hf_std:.2f}")
        print(f"Base threshold (level 1 before gamma scaling): {base_threshold:.2f}")
        print(f"Per-level gamma: {LEVEL_THRESHOLD_GAMMA}")

        total_coeffs = 0
        total_significant = 0

        # Extract sparse 2D coordinates for each band at each level
        for level_idx in range(1, len(coeffs)):
            cH, cV, cD = coeffs[level_idx]

            # Compute per-level threshold (coeffs[1] is finest; higher indices are coarser)
            level_threshold = base_threshold * (LEVEL_THRESHOLD_GAMMA ** (level_idx - 1))

            print(f"\nLevel {level_idx}: (threshold={level_threshold:.2f})")
            print(f"  cH shape: {cH.shape}, cV shape: {cV.shape}, cD shape: {cD.shape}")

            # Extract sparse data for each band with per-level threshold
            cH_vals, cH_coords, cH_rows, cH_cols, cH_count = extract_2d_sparse_coeffs(cH, level_threshold)
            cV_vals, cV_coords, cV_rows, cV_cols, cV_count = extract_2d_sparse_coeffs(cV, level_threshold)
            cD_vals, cD_coords, cD_rows, cD_cols, cD_count = extract_2d_sparse_coeffs(cD, level_threshold)

            level_total = cH.size + cV.size + cD.size
            level_significant = cH_count + cV_count + cD_count
            total_coeffs += level_total
            total_significant += level_significant

            print(f"  Significant: cH={cH_count}/{cH.size}, cV={cV_count}/{cV.size}, cD={cD_count}/{cD.size}")
            print(f"  Level sparsity: {100*(1-level_significant/level_total):.2f}%")

            # Store sparse data with indices for reconstruction
            band_data_per_level.append({
                'cH': {'values': cH_vals, 'coords': cH_coords, 'rows': cH_rows, 'cols': cH_cols, 'shape': cH.shape},
                'cV': {'values': cV_vals, 'coords': cV_coords, 'rows': cV_rows, 'cols': cV_cols, 'shape': cV.shape},
                'cD': {'values': cD_vals, 'coords': cD_coords, 'rows': cD_rows, 'cols': cD_cols, 'shape': cD.shape},
                'threshold': level_threshold
            })

        print(f"\nOverall sparsity: {100*(1-total_significant/total_coeffs):.2f}%")
        print(f"Total significant coefficients: {total_significant}/{total_coeffs}")
    else:
        # LL-only mode - just store shapes
        for level_idx in range(1, len(coeffs)):
            cH, cV, cD = coeffs[level_idx]
            band_data_per_level.append({
                'cH': {'values': np.array([]), 'coords': np.array([]), 'shape': cH.shape},
                'cV': {'values': np.array([]), 'coords': np.array([]), 'shape': cV.shape},
                'cD': {'values': np.array([]), 'coords': np.array([]), 'shape': cD.shape}
            })
        print("High-frequency coefficients will be filled with zeros (LL-only mode)")

    # Setup CUDA and data types
    dtype = torch.float32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

    # Set random seed for reproducibility
    seed = random.randint(1, int(1e6))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Initialize results dictionary
    results = {
        'fp_bpp': [], 'hp_bpp': [], 
        'fp_psnr': [], 'hp_psnr': [], 
        'fp_calc_psnr': [], 'hp_calc_psnr': []
    }

    # Create output directory
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Normalize LL coefficients
    ll_mean = ll_coeffs.mean()
    ll_std = ll_coeffs.std()
    ll_norm = (ll_coeffs - ll_mean) / (ll_std + 1e-8)
    
    print(f"\nLL coefficient statistics:")
    print(f"  Mean: {ll_mean:.2f}, Std: {ll_std:.2f}")
    print(f"  Min: {ll_coeffs.min():.2f}, Max: {ll_coeffs.max():.2f}")
    print(f"  Normalized min: {ll_norm.min():.2f}, max: {ll_norm.max():.2f}")
    
    # Convert normalized LL coefficients to tensor
    coeffs_tensor = transforms.ToTensor()(ll_norm).float().to(device, dtype)

    # Adaptively scale LL model size based on coefficient count
    ll_h, ll_w = ll_coeffs.shape
    ll_pixels = ll_h * ll_w
    
    # Formula-based adaptive sizing: scale with log of coefficient count
    # More coefficients = larger model capacity needed
    # Formula: layers = max(3, min(max_layers, a * log(coeffs) + b))
    #          size = max(32, min(max_size, c * log(coeffs) + d))
    import math
    
    log_pixels = math.log10(max(ll_pixels, 10))  # log base 10, min 10 to avoid log(0)
    
    # Layers scale: ~3 for 100 pixels, ~5 for 500, ~8 for 2000, ~12 for 5000+
    adaptive_ll_layers = max(3, min(MODEL_NUM_LAYERS, int(3.5 * log_pixels - 4)))
    
    # Size scales: ~32 for 100 pixels, ~56 for 500, ~72 for 2000, ~96 for 5000+
    base_ll_size = max(32, min(MODEL_LAYER_SIZE, int(32 * log_pixels - 32)))
    # Apply LL-specific width scale for better quality
    adaptive_ll_size = max(16, int(base_ll_size * MODEL_SIZE_SCALE_LL))
    
    # More iterations for smaller models (they converge faster but need more epochs)
    adaptive_ll_iters = 2500 if ll_pixels < 1000 else (2000 if ll_pixels < 3000 else MODEL_ITERATIONS)
    
    print(f"\nAdaptive LL model: {adaptive_ll_layers} layers × {adaptive_ll_size} units ({adaptive_ll_iters} iters)")
    print(f"  (for {ll_pixels} LL coefficients, log={log_pixels:.2f})")

    # Initialize SIREN model for LL coefficients
    func_rep_ll = Siren(
        dim_in=2,
        dim_hidden=adaptive_ll_size,
        dim_out=1,
        num_layers=adaptive_ll_layers,
        final_activation=torch.nn.Identity(),
        w0_initial=30.0,
        w0=30.0
    )
    
    # Helper function to get adaptive HF model parameters
    def get_adaptive_hf_params(num_coeffs):
        """Adaptively scale HF model based on number of significant coefficients using formula"""
        log_coeffs = math.log10(max(num_coeffs, 10))
        
        # Layers: ~3 for 100 coeffs, ~5 for 500, ~8 for 2000, ~10 for 10k, ~13 for 30k+
        layers = max(3, min(MODEL_NUM_LAYERS, int(3.5 * log_coeffs - 4)))
        
        # Size: ~32 for 100 coeffs, ~48 for 500, ~64 for 2000, ~80 for 10k, ~96 for 30k+
        base_size = max(32, min(MODEL_LAYER_SIZE, int(32 * log_coeffs - 32)))
        # Apply HF-specific width scale (smaller for compression)
        size = max(16, int(base_size * MODEL_SIZE_SCALE_HF))
        
        # Iterations: more for smaller datasets
        iters = 2500 if num_coeffs < 1000 else (2000 if num_coeffs < 5000 else (1500 if num_coeffs < 20000 else MODEL_ITERATIONS))
        
        return layers, size, iters
    
    # Initialize separate SIREN models for each band at each level
    band_models = []
    
    if USE_SPARSE_HF:
        for level_idx, level_data in enumerate(band_data_per_level):
            level_models = {}
            
            for band_name in ['cH', 'cV', 'cD']:
                if len(level_data[band_name]['values']) > 0:
                    num_coeffs = len(level_data[band_name]['values'])
                    
                    # Get adaptive parameters based on coefficient count
                    hf_layers, hf_size, hf_iters = get_adaptive_hf_params(num_coeffs)
                    
                    model = Siren(
                        dim_in=2,  # 2D coordinates (x, y)
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

    # Train LL model
    trainer_ll = Trainer(func_rep_ll, lr=2e-4)
    coordinates_ll, features_ll = util.to_coordinates_and_coeffs_features(coeffs_tensor)
    coordinates_ll, features_ll = coordinates_ll.to(device, dtype), features_ll.to(device, dtype)

    print("\n=== Training LL (Low-frequency) Model ===")
    print(f"Training data shape: coords={coordinates_ll.shape}, features={features_ll.shape}")
    print(f"Coordinates range: [{coordinates_ll.min():.2f}, {coordinates_ll.max():.2f}]")
    print(f"Features range: [{features_ll.min():.2f}, {features_ll.max():.2f}]")
    print(f"Features mean: {features_ll.mean():.4f}, std: {features_ll.std():.4f}")
    
    trainer_ll.train(coordinates_ll, features_ll, num_iters=adaptive_ll_iters)
    print(f'Best LL training PSNR: {trainer_ll.best_vals["psnr"]:.2f}')
    
    # Train HF models for each band at each level
    if USE_SPARSE_HF:
        for level_idx, level_models in enumerate(band_models):
            print(f"\n=== Training Level {level_idx+1} High-frequency Models ===")
            
            for band_name in ['cH', 'cV', 'cD']:
                if band_name in level_models:
                    band_info = level_models[band_name]
                    num_coeffs = len(band_info['values'])
                    
                    # Print adaptive model configuration
                    print(f"  {band_name}: {band_info['adaptive_layers']}L×{band_info['adaptive_size']}U, {band_info['adaptive_iters']} iters ({num_coeffs} coeffs)")
                    
                    # Normalize values
                    values = band_info['values']
                    val_mean = values.mean()
                    val_std = values.std()
                    val_norm = (values - val_mean) / (val_std + 1e-8)
                    
                    # Store normalization params for reconstruction
                    band_info['mean'] = val_mean
                    band_info['std'] = val_std
                    
                    # Prepare training data
                    coords_tensor = torch.tensor(band_info['coords'], dtype=torch.float32).to(device, dtype)
                    features_tensor = torch.tensor(val_norm.reshape(-1, 1), dtype=torch.float32).to(device, dtype)
                    
                    # Train with adaptive iterations
                    trainer = Trainer(band_info['model'], lr=2e-4)
                    trainer.train(coords_tensor, features_tensor, num_iters=band_info['adaptive_iters'])
                    
                    print(f"    → PSNR={trainer.best_vals['psnr']:.2f}")
                    
                    # Store trainer for saving best model
                    band_info['trainer'] = trainer

    # Calculate and log model size metrics
    model_size_ll = util.model_size_in_bits(func_rep_ll) / 8000.
    print(f'\nLL Model size: {model_size_ll:.1f}kB')
    print(f'  Architecture: {adaptive_ll_layers}L × {adaptive_ll_size}U')
    
    total_model_size = model_size_ll
    
    if USE_SPARSE_HF and band_models:
        print(f'\nHF Models per band:')
        total_hf_size = 0
        for level_idx, level_models in enumerate(band_models):
            print(f'  Level {level_idx+1}:')
            for band_name, band_info in level_models.items():
                band_size = util.model_size_in_bits(band_info['model']) / 8000.
                total_hf_size += band_size
                print(f'    {band_name}: {band_size:.1f}kB ({band_info["adaptive_layers"]}L × {band_info["adaptive_size"]}U)')
        
        print(f'HF Models total size: {total_hf_size:.1f}kB')
        total_model_size += total_hf_size
        print(f'Total Model size (FP32): {total_model_size:.1f}kB')
        print(f'Total Model size (FP16): {total_model_size/2:.1f}kB')
    
    fp_bpp = util.bpp(model=func_rep_ll, image=coeffs_tensor)
    print(f'Full precision bpp (LL): {fp_bpp:.2f}')

    # Log full precision results
    results['fp_bpp'].append(fp_bpp)
    results['fp_psnr'].append(trainer_ll.best_vals['psnr'])

    # Save best models
    torch.save(trainer_ll.best_model, os.path.join(LOG_DIR, 'best_model_ll.pt'))
    
    if USE_SPARSE_HF:
        for level_idx, level_models in enumerate(band_models):
            for band_name, band_info in level_models.items():
                model_path = os.path.join(LOG_DIR, f'best_model_level{level_idx+1}_{band_name}.pt')
                torch.save(band_info['trainer'].best_model, model_path)

    # Load best models and convert to half precision
    func_rep_ll.load_state_dict(trainer_ll.best_model)
    func_rep_ll = func_rep_ll.half().to('cuda')
    coordinates_ll = coordinates_ll.half().to('cuda')
    
    if USE_SPARSE_HF:
        for level_models in band_models:
            for band_name, band_info in level_models.items():
                band_info['model'].load_state_dict(band_info['trainer'].best_model)
                band_info['model'] = band_info['model'].half().to('cuda')

    # Calculate half precision metrics
    hp_bpp = util.bpp(model=func_rep_ll, image=coeffs_tensor)
    results['hp_bpp'].append(hp_bpp)
    print(f'\nHalf precision bpp: {hp_bpp:.2f}')

    with torch.no_grad():
        # Reconstruct LL coefficients
        ll_recon = func_rep_ll(coordinates_ll).reshape(coeffs_tensor.shape[1], coeffs_tensor.shape[2]).float()
        ll_recon_denorm = ll_recon * (ll_std + 1e-8) + ll_mean
        
        # Calculate PSNR on LL coefficients
        original_ll_coeffs = torch.tensor(ll_coeffs).to(device, dtype)
        hp_psnr = util.get_clamped_psnr(original_ll_coeffs, ll_recon_denorm)
        
        # Save metrics
        results_file = os.path.join(LOG_DIR, 'metrics.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Convert reconstructed LL coefficients to numpy
        ll_recon_np = ll_recon_denorm.cpu().numpy()
        
        # Create coefficient structure for reconstruction
        coeffs_modified = [ll_recon_np]
        
        # Reconstruct high-frequency bands
        if USE_SPARSE_HF:
            for level_idx, level_models in enumerate(band_models):
                cH_recon = np.zeros(band_data_per_level[level_idx]['cH']['shape'])
                cV_recon = np.zeros(band_data_per_level[level_idx]['cV']['shape'])
                cD_recon = np.zeros(band_data_per_level[level_idx]['cD']['shape'])
                
                # Reconstruct each band
                for band_name, band_array in [('cH', cH_recon), ('cV', cV_recon), ('cD', cD_recon)]:
                    if band_name in level_models:
                        band_info = level_models[band_name]
                        h, w = band_info['shape']
                        
                        # Get the trained positions (rows, cols)
                        trained_coords = band_info['coords']
                        trained_rows = band_info['rows']
                        trained_cols = band_info['cols']
                        
                        # Predict ONLY for trained positions
                        trained_coords_tensor = torch.tensor(trained_coords, dtype=torch.float32).to(device).half()
                        predictions = band_info['model'](trained_coords_tensor).reshape(-1).float()
                        
                        # Denormalize
                        predictions_denorm = predictions * (band_info['std'] + 1e-8) + band_info['mean']
                        predictions_np = predictions_denorm.cpu().numpy()
                        
                        # Fill ONLY trained positions, rest stays 0
                        band_array[trained_rows, trained_cols] = predictions_np
                
                coeffs_modified.append((cH_recon, cV_recon, cD_recon))
        else:
            # Zero-filled mode
            for level_data in band_data_per_level:
                cH_zero = np.zeros(level_data['cH']['shape'])
                cV_zero = np.zeros(level_data['cV']['shape'])
                cD_zero = np.zeros(level_data['cD']['shape'])
                coeffs_modified.append((cH_zero, cV_zero, cD_zero))
        
        coeffs_recon_list = coeffs_modified
        
        # Perform inverse DWT
        img_recon = pywt.waverec2(coeffs_recon_list, wavelet=WAVELET)
        
        # Save reconstructed image
        img_recon = np.clip(img_recon, 0, 255).astype(np.uint8)
        out_img = Image.fromarray(img_recon)
        out_img.save(OUTPUT_FILE)
        
        # Calculate and print image-space PSNR
        original_img = np.asarray(Image.open(f"kodak-dataset/{IMAGEID}.png").convert("L"))
        img_psnr = util.calc_psnr(original_img, img_recon)
        print(f'Image-space PSNR: {img_psnr:.2f}')
        
        print(f'Half precision PSNR: {hp_psnr:.2f}')
        results['hp_psnr'].append(hp_psnr)
        results['hp_calc_psnr'].append(float(img_psnr))

    # Calculate total running time
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print compression comparison
    print("\n" + "="*60)
    print("COMPRESSION SUMMARY")
    print("="*60)
    print(f"Original image size: {original_size_kb:.2f} kB ({original_bpp:.2f} bpp)")
    print(f"Model size (FP32):   {total_model_size:.2f} kB")
    print(f"Model size (FP16):   {total_model_size/2:.2f} kB")
    print(f"Compression ratio:   {original_size_kb/(total_model_size/2):.2f}x")
    print(f"Image PSNR:          {img_psnr:.2f} dB")
    print(f"Total running time:  {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("="*60)
    
    print("\n2D Bands Results:", results)

if __name__ == "__main__":
    main()
