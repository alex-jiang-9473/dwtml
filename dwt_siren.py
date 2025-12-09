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

# ---------------------------
# CONFIG
IMAGEID = "kodim01"  # Image to compress
LEVELS = 1        # Number of DWT decomposition levels
WAVELET = "db4"   # Wavelet type: 'db1','db4','sym4', etc.
LOG_DIR = "results/dwt"
NUM_LAYERS = 10   # Number of layers in SIREN
LAYER_SIZE = 56   # Hidden layer size
ITERATIONS = 5000 # Training iterations
OUTPUT_FILE = os.path.join(LOG_DIR, f"{IMAGEID}_dwt_siren_L{NUM_LAYERS}_S{LAYER_SIZE}_I{ITERATIONS}.png")

# ---------------------------

def main():
    import time
    start_time = time.time()
    
    # 1) Load and prepare grayscale image
    img = Image.open(f"kodak-dataset/{IMAGEID}.png").convert("L")
    A = np.asarray(img, dtype=np.float32)

    # 2) Perform multi-level 2D DWT
    coeffs = pywt.wavedec2(A, wavelet=WAVELET, level=LEVELS)

    # 3) Convert coefficients to array and normalize
    arr, slices = pywt.coeffs_to_array(coeffs)

    # Save coefficients for further analysis
    coeffs_file = os.path.join(LOG_DIR, f'{IMAGEID}_coeffs_L{LEVELS}_{WAVELET}.npz')
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Convert slices to a saveable format
    slices_list = [str(s) for s in slices]  # Convert slices to strings
    
    np.savez_compressed(
        coeffs_file,
        coeffs_array=arr,
        slices_str=slices_list,  # Save as strings
        image_shape=A.shape,
        wavelet=WAVELET,
        levels=LEVELS
    )
    print(f"Coefficients saved to: {coeffs_file}")

    # Normalize coefficients for training stability
    arr_mean = arr.mean()
    arr_std = arr.std()
    arr_norm = (arr - arr_mean) / (arr_std + 1e-8)

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
    
    # Convert normalized coefficients to tensor
    coeffs_tensor = transforms.ToTensor()(arr_norm).float().to(device, dtype)

    # Initialize SIREN model
    func_rep = Siren(
        dim_in=2,              # Input dimension (x,y coordinates)
        dim_hidden=LAYER_SIZE, # Hidden layer size
        dim_out=1,            # Output dimension (coefficient value)
        num_layers=NUM_LAYERS,
        final_activation=torch.nn.Identity(),
        w0_initial=30.0,      # Frequency scaling for first layer
        w0=30.0              # Frequency scaling for other layers
    )

    # Initialize trainer
    trainer = Trainer(func_rep, lr=2e-4)

    # Generate coordinates and features for training
    coordinates, features = util.to_coordinates_and_coeffs_features(coeffs_tensor)
    coordinates, features = coordinates.to(device, dtype), features.to(device, dtype)

    # Calculate and log model size metrics
    model_size = util.model_size_in_bits(func_rep) / 8000.
    print(f'Model size: {model_size:.1f}kB')
    fp_bpp = util.bpp(model=func_rep, image=coeffs_tensor)
    print(f'Full precision bpp: {fp_bpp:.2f}')

    # Train the model
    train_start = time.time()
    trainer.train(coordinates, features, num_iters=ITERATIONS)
    train_end = time.time()
    training_time = train_end - train_start
    print(f'Best training psnr: {trainer.best_vals["psnr"]:.2f}')
    print(f'Training time: {training_time:.2f} seconds')

    # Log full precision results
    results['fp_bpp'].append(fp_bpp)
    results['fp_psnr'].append(trainer.best_vals['psnr'])

    # Save best model
    best_model_file = f'best_model_dwt_L{NUM_LAYERS}_S{LAYER_SIZE}_I{ITERATIONS}.pt'
    torch.save(trainer.best_model, os.path.join(LOG_DIR, best_model_file))

    # Load best model and convert to half precision
    func_rep.load_state_dict(trainer.best_model)
    func_rep = func_rep.half().to('cuda')
    coordinates = coordinates.half().to('cuda')

    # Calculate half precision metrics
    hp_bpp = util.bpp(model=func_rep, image=coeffs_tensor)
    results['hp_bpp'].append(hp_bpp)
    print(f'Half precision bpp: {hp_bpp:.2f}')

    with torch.no_grad():
        # Reconstruct coefficients from network
        coeffs_recon = func_rep(coordinates).reshape(coeffs_tensor.shape[1], coeffs_tensor.shape[2]).float()
        
        # Denormalize reconstructed coefficients
        coeffs_recon_denorm = coeffs_recon * (arr_std + 1e-8) + arr_mean
        
        # Calculate PSNR on coefficients
        original_coeffs = torch.tensor(arr).to(device, dtype)
        hp_psnr = util.get_clamped_psnr(original_coeffs, coeffs_recon_denorm)
        
        # Convert reconstructed coefficients for inverse DWT
        coeffs_np = coeffs_recon_denorm.cpu().numpy()
        coeffs_recon_list = pywt.array_to_coeffs(coeffs_np, slices, output_format='wavedec2')
        
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
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in func_rep.parameters())
        
        # Get GPU memory usage (if available)
        gpu_memory_mb = 0
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
            torch.cuda.reset_peak_memory_stats()
        
        # Calculate total execution time
        total_time = time.time() - start_time
        
        # Compile comprehensive results for plotting
        plot_results = {
            'config': {
                'image_id': IMAGEID,
                'num_layers': NUM_LAYERS,
                'layer_size': LAYER_SIZE,
                'iterations': ITERATIONS,
                'wavelet': WAVELET,
                'dwt_levels': LEVELS
            },
            'image': {
                'original_shape': list(A.shape),
                'original_size_bytes': int(A.size),
                'original_size_kb': float(A.size / 1024.0)
            },
            'coefficients': {
                'total_count': int(arr.size),
                'shape': list(arr.shape),
                'min': float(arr.min()),
                'max': float(arr.max()),
                'mean': float(arr.mean()),
                'std': float(arr.std())
            },
            'model': {
                'total_parameters': total_params,
                'model_size_kb': model_size,
                'model_architecture': f"{NUM_LAYERS}×{LAYER_SIZE}"
            },
            'compression': {
                'fp32_bpp': fp_bpp,
                'fp16_bpp': hp_bpp,
                'original_size_kb': A.size / 1024.0,
                'compressed_size_kb': model_size,
                'compression_ratio': (A.size / 1024.0) / model_size
            },
            'quality': {
                'fp32_psnr': trainer.best_vals['psnr'],
                'fp16_psnr': float(hp_psnr),
                'image_space_psnr': float(img_psnr),
                'best_training_loss': trainer.best_vals.get('loss', None)
            },
            'training': {
                'iterations': ITERATIONS,
                'learning_rate': 2e-4,
                'seed': seed,
                'training_time_sec': float(training_time),
                'total_time_sec': float(total_time)
            },
            'hardware': {
                'gpu_memory_mb': float(gpu_memory_mb),
                'device': str(device)
            }
        }
        
        # Save comprehensive results
        results_file = os.path.join(LOG_DIR, f'{IMAGEID}_results_L{NUM_LAYERS}_S{LAYER_SIZE}_I{ITERATIONS}.json')
        with open(results_file, 'w') as f:
            json.dump(plot_results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        print(f"Model: {NUM_LAYERS} layers × {LAYER_SIZE} units = {total_params:,} params")
        print(f"Compression: {plot_results['compression']['compression_ratio']:.2f}x")
        print(f"Total time: {total_time:.2f} seconds")
        if gpu_memory_mb > 0:
            print(f"Peak GPU memory: {gpu_memory_mb:.2f} MB")

    print("DWT Results:", results)

if __name__ == "__main__":
    main()