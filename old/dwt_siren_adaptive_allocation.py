import os
import random
import json
import numpy as np
import pywt
import torch
from PIL import Image
from torchvision import transforms
from training import Trainer
from siren import Siren
import util
import time

# ---------------------------
# CONFIG
IMAGEID = "kodim01"  # Image to compress
LEVELS = 1        # Number of DWT decomposition levels (only 1 level supported)
WAVELET = "db4"   # Wavelet type: 'db1','db4','sym4', etc.
ITERATIONS = 2000 # Training iterations for each band

# Allocation strategy identifier (used in output filenames)
ALLOCATION_STRATEGY = "more_ll"  # Options: "even", "more_ll", "more_hf"
LOG_DIR = f"results/dwt_adaptive_{ALLOCATION_STRATEGY}"

# Model architecture for each band (layers × neurons)
# Adjust these to allocate more/less capacity to different frequency bands

# OPTION 1: Even distribution (all bands equal)
# BAND_ARCHITECTURES = {
#     'LL': {'layers': 5, 'neurons': 40},  # ~8,000 params
#     'LH': {'layers': 5, 'neurons': 40},  # ~8,000 params
#     'HL': {'layers': 5, 'neurons': 40},  # ~8,000 params
#     'HH': {'layers': 5, 'neurons': 40}   # ~8,000 params
# }
# Total: ~32,000 params

# OPTION 2: More LL (emphasize low-frequency approximation)
BAND_ARCHITECTURES = {
    'LL': {'layers': 8, 'neurons': 56},  # ~25,088 params
    'LH': {'layers': 4, 'neurons': 24},  # ~2,304 params
    'HL': {'layers': 4, 'neurons': 24},  # ~2,304 params
    'HH': {'layers': 4, 'neurons': 24}   # ~2,304 params
}
# Total: ~32,000 params

# OPTION 3: More HF (emphasize high-frequency details)
# BAND_ARCHITECTURES = {
#     'LL': {'layers': 3, 'neurons': 24},   # ~1,728 params
#     'LH': {'layers': 6, 'neurons': 48},   # ~13,824 params
#     'HL': {'layers': 6, 'neurons': 48},   # ~13,824 params
#     'HH': {'layers': 3, 'neurons': 32}    # ~3,072 params
# }
# Total: ~32,448 params

# Learning rate (fixed for all bands)
LEARNING_RATE = 2e-4

# ---------------------------

def train_band(band_name, coeffs, band_slice, num_layers, layer_size, lr, iterations, device, dtype):
    """
    Train a SIREN model for a specific frequency band.
    
    Args:
        band_name: Name of the band (LL, LH, HL, HH)
        coeffs: Full coefficient array
        band_slice: Tuple of slices to extract this band
        num_layers: Number of layers for this band's model
        layer_size: Hidden layer size for this band's model
        lr: Learning rate
        iterations: Training iterations
        device: PyTorch device
        dtype: PyTorch data type
    
    Returns:
        Dictionary with trained model, statistics, and reconstruction
    """
    print(f"\n{'='*60}")
    print(f"Training {band_name} band")
    print(f"Architecture: {num_layers} layers × {layer_size} neurons")
    print(f"Learning rate: {lr}")
    print(f"{'='*60}")
    
    # Extract band coefficients
    band_coeffs = coeffs[band_slice]
    
    # Normalize
    band_mean = band_coeffs.mean()
    band_std = band_coeffs.std()
    band_norm = (band_coeffs - band_mean) / (band_std + 1e-8)
    
    # Convert to tensor
    band_tensor = transforms.ToTensor()(band_norm).float().to(device, dtype)
    
    # Initialize SIREN model
    func_rep = Siren(
        dim_in=2,
        dim_hidden=layer_size,
        dim_out=1,
        num_layers=num_layers,
        final_activation=torch.nn.Identity(),
        w0_initial=30.0,
        w0=30.0
    )
    
    # Initialize trainer
    trainer = Trainer(func_rep, lr=lr)
    
    # Generate coordinates and features
    coordinates, features = util.to_coordinates_and_coeffs_features(band_tensor)
    coordinates, features = coordinates.to(device, dtype), features.to(device, dtype)
    
    # Calculate model size
    model_size_kb = util.model_size_in_bits(func_rep) / 8000.
    total_params = sum(p.numel() for p in func_rep.parameters())
    
    print(f"Band shape: {band_coeffs.shape}")
    print(f"Band stats: mean={band_mean:.2f}, std={band_std:.2f}")
    print(f"Model size: {model_size_kb:.2f}kB, {total_params:,} params")
    
    # Train
    train_start = time.time()
    trainer.train(coordinates, features, num_iters=iterations)
    train_time = time.time() - train_start
    
    print(f"Training PSNR: {trainer.best_vals['psnr']:.2f} dB")
    print(f"Training time: {train_time:.2f}s")
    
    # Load best model and convert to half precision
    func_rep.load_state_dict(trainer.best_model)
    func_rep = func_rep.half().to(device)
    coordinates = coordinates.half().to(device)
    
    # Reconstruct band
    with torch.no_grad():
        band_recon = func_rep(coordinates).reshape(band_tensor.shape[1], band_tensor.shape[2]).float()
        band_recon_denorm = band_recon * (band_std + 1e-8) + band_mean
        band_recon_np = band_recon_denorm.cpu().numpy()
    
    # Calculate metrics
    hp_bpp = util.bpp(model=func_rep, image=band_tensor)
    original_band_tensor = torch.tensor(band_coeffs).to(device, dtype)
    hp_psnr = util.get_clamped_psnr(original_band_tensor, band_recon_denorm)
    
    return {
        'model_state': trainer.best_model,
        'model': func_rep,
        'reconstruction': band_recon_np,
        'normalization': {'mean': band_mean, 'std': band_std},
        'shape': band_coeffs.shape,
        'slice': band_slice,
        'stats': {
            'num_layers': num_layers,
            'layer_size': layer_size,
            'total_params': total_params,
            'model_size_kb': model_size_kb,
            'hp_bpp': hp_bpp,
            'hp_psnr': float(hp_psnr),
            'training_psnr': trainer.best_vals['psnr'],
            'training_time_sec': train_time,
            'learning_rate': lr,
            'iterations': iterations
        }
    }

def main():
    start_time = time.time()
    
    # 1) Load and prepare grayscale image
    img = Image.open(f"kodak-dataset/{IMAGEID}.png").convert("L")
    A = np.asarray(img, dtype=np.float32)
    print(f"Image shape: {A.shape}")
    
    # 2) Perform single-level 2D DWT using wavedec2 for consistency
    coeffs = pywt.wavedec2(A, wavelet=WAVELET, level=LEVELS)
    
    # 3) Create full coefficient array and slices for reconstruction
    arr, slices = pywt.coeffs_to_array(coeffs)
    
    # Extract band information for display
    if LEVELS == 1:
        LL = coeffs[0]
        LH, HL, HH = coeffs[1]
        print(f"\nBand shapes:")
        print(f"  LL: {LL.shape}")
        print(f"  LH: {LH.shape}")
        print(f"  HL: {HL.shape}")
        print(f"  HH: {HH.shape}")
    
    print(f"\nFull coefficient array shape: {arr.shape}")
    print(f"Slices: {slices}")
    
    # 4) Setup CUDA
    dtype = torch.float32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')
    
    # Set random seed
    seed = random.randint(1, int(1e6))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 5) Print architecture for each band
    print(f"\n{'='*60}")
    print("MODEL ALLOCATION")
    print(f"{'='*60}")
    
    total_allocated_params = 0
    
    for band_name, arch in BAND_ARCHITECTURES.items():
        num_layers = arch['layers']
        layer_size = arch['neurons']
        
        # Rough parameter count estimate
        approx_params = num_layers * layer_size * layer_size
        total_allocated_params += approx_params
        
        print(f"{band_name}: {num_layers}L × {layer_size}N (~{approx_params:,} params)")
    
    print(f"Total estimated params: ~{total_allocated_params:,}")
    
    # 6) Create output directory
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 7) Train each band separately
    trained_bands = {}
    
    # Extract band slices from the slices object
    # For single-level wavedec2: slices = [LL_slice, {'ad': LH_slice, 'da': HL_slice, 'dd': HH_slice}]
    ll_slice = slices[0]
    detail_slices = slices[1]
    
    band_slices = {
        'LL': ll_slice,
        'LH': detail_slices['ad'],  # LH = horizontal (ad)
        'HL': detail_slices['da'],  # HL = vertical (da)
        'HH': detail_slices['dd']   # HH = diagonal (dd)
    }
    
    for band_name in ['LL', 'LH', 'HL', 'HH']:
        num_layers = BAND_ARCHITECTURES[band_name]['layers']
        layer_size = BAND_ARCHITECTURES[band_name]['neurons']
        band_slice = band_slices[band_name]
        
        trained_bands[band_name] = train_band(
            band_name, arr, band_slice, num_layers, layer_size, 
            LEARNING_RATE, ITERATIONS, device, dtype
        )
        
        # Save individual band model
        model_file = os.path.join(LOG_DIR, f'best_model_{band_name}.pt')
        torch.save(trained_bands[band_name]['model_state'], model_file)
    
    # 8) Reconstruct full coefficient array
    print(f"\n{'='*60}")
    print("RECONSTRUCTING FULL IMAGE")
    print(f"{'='*60}")
    
    arr_recon = np.zeros_like(arr)
    for band_name in ['LL', 'LH', 'HL', 'HH']:
        band_slice = band_slices[band_name]
        arr_recon[band_slice] = trained_bands[band_name]['reconstruction']
    
    # 9) Convert back to image
    coeffs_recon_list = pywt.array_to_coeffs(arr_recon, slices, output_format='wavedec2')
    img_recon = pywt.waverec2(coeffs_recon_list, wavelet=WAVELET)
    
    # 10) Save reconstructed image
    img_recon = np.clip(img_recon, 0, 255).astype(np.uint8)
    out_img = Image.fromarray(img_recon)
    output_file = os.path.join(LOG_DIR, f"{IMAGEID}_{ALLOCATION_STRATEGY}.png")
    out_img.save(output_file)
    
    # 11) Calculate image-space PSNR
    img_psnr = util.calc_psnr(A, img_recon)
    print(f"\nFinal Image-space PSNR: {img_psnr:.2f} dB")
    
    # 12) Calculate total metrics
    total_params = sum(band['stats']['total_params'] for band in trained_bands.values())
    total_model_size_kb = sum(band['stats']['model_size_kb'] for band in trained_bands.values())
    total_training_time = sum(band['stats']['training_time_sec'] for band in trained_bands.values())
    
    # GPU memory
    gpu_memory_mb = 0
    if torch.cuda.is_available():
        gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
        torch.cuda.reset_peak_memory_stats()
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("FINAL STATISTICS")
    print(f"{'='*60}")
    print(f"Total parameters: {total_params:,}")
    print(f"Total model size: {total_model_size_kb:.2f} kB")
    print(f"Compression ratio: {(A.size / 1024.0) / total_model_size_kb:.2f}x")
    print(f"Total training time: {total_training_time:.2f}s")
    print(f"Total execution time: {total_time:.2f}s")
    if gpu_memory_mb > 0:
        print(f"Peak GPU memory: {gpu_memory_mb:.2f} MB")
    
    # 13) Save comprehensive results
    results = {
        'config': {
            'image_id': IMAGEID,
            'allocation_strategy': ALLOCATION_STRATEGY,
            'wavelet': WAVELET,
            'dwt_levels': LEVELS,
            'band_architectures': BAND_ARCHITECTURES,
            'learning_rate': LEARNING_RATE,
            'iterations': ITERATIONS
        },
        'image': {
            'original_shape': list(A.shape),
            'original_size_kb': float(A.size / 1024.0)
        },
        'bands': {
            band_name: {
                'architecture': f"{band['stats']['num_layers']}×{band['stats']['layer_size']}",
                'shape': list(band['shape']),
                'params': band['stats']['total_params'],
                'model_size_kb': band['stats']['model_size_kb'],
                'bpp': band['stats']['hp_bpp'],
                'psnr': band['stats']['hp_psnr'],
                'training_psnr': band['stats']['training_psnr'],
                'training_time_sec': band['stats']['training_time_sec'],
                'learning_rate': band['stats']['learning_rate']
            }
            for band_name, band in trained_bands.items()
        },
        'overall': {
            'total_parameters': total_params,
            'total_model_size_kb': total_model_size_kb,
            'compression_ratio': (A.size / 1024.0) / total_model_size_kb,
            'image_space_psnr': float(img_psnr),
            'total_training_time_sec': total_training_time,
            'total_execution_time_sec': total_time,
            'gpu_memory_mb': float(gpu_memory_mb)
        }
    }
    
    results_file = os.path.join(LOG_DIR, f'{IMAGEID}_{ALLOCATION_STRATEGY}_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Reconstructed image saved to: {output_file}")
    
    # Print per-band breakdown
    print(f"\n{'='*60}")
    print("PER-BAND BREAKDOWN")
    print(f"{'='*60}")
    for band_name in ['LL', 'LH', 'HL', 'HH']:
        band = trained_bands[band_name]
        print(f"\n{band_name}:")
        print(f"  Architecture: {band['stats']['num_layers']}×{band['stats']['layer_size']}")
        print(f"  Parameters: {band['stats']['total_params']:,}")
        print(f"  Model size: {band['stats']['model_size_kb']:.2f} kB")
        print(f"  PSNR: {band['stats']['hp_psnr']:.2f} dB")
        print(f"  Training time: {band['stats']['training_time_sec']:.2f}s")

if __name__ == "__main__":
    main()
