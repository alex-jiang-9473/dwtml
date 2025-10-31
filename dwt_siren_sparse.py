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
LEVELS = 3        # Number of DWT decomposition levels
WAVELET = "db4"   # Wavelet type: 'db1','db4','sym4', etc.
LOG_DIR = "results/dwt_sparse"

# LL Model Configuration
LL_NUM_LAYERS = 10   # Number of layers in LL SIREN
LL_LAYER_SIZE = 28   # Hidden layer size for LL
LL_ITERATIONS = 1000 # Training iterations for LL

# High-frequency Model Configuration
HF_NUM_LAYERS = 10   # Number of layers in HF SIREN (can be different from LL)
HF_LAYER_SIZE = 28   # Hidden layer size for HF (can be smaller/larger than LL)
HF_ITERATIONS = 1000 # Training iterations for HF (can be more/less than LL)

# High-frequency coefficient handling mode
USE_SPARSE_HF = True  # True: Use sparse threshold training for HF, False: Fill HF with zeros
THRESHOLD_FACTOR = 0.05  # Only used when USE_SPARSE_HF=True. Keep coefficients above this * std
# Range: 0.01 - 2.00. Lower values keep more coefficients (less sparse)

OUTPUT_FILE = f"dwt_siren_{'sparse' if USE_SPARSE_HF else 'LL_only'}_levels{LEVELS}_{WAVELET}_LL{LL_NUM_LAYERS}x{LL_LAYER_SIZE}_HF{HF_NUM_LAYERS}x{HF_LAYER_SIZE}_{LL_ITERATIONS}.png"

# ---------------------------

def main():
    # 1) Load and prepare grayscale image
    img = Image.open("kodak-dataset/kodim01.png").convert("L")
    A = np.asarray(img, dtype=np.float32)

    # 2) Perform multi-level 2D DWT
    coeffs = pywt.wavedec2(A, wavelet=WAVELET, level=LEVELS)

    # 3) Extract only the LL (approximation) coefficients
    # coeffs format: [cA_n, (cH_n, cV_n, cD_n), (cH_n-1, cV_n-1, cD_n-1), ...]
    # We only want cA_n (the first element - LL part)
    ll_coeffs = coeffs[0]  # This is the top-left corner (approximation)
    
    print(f"Original image shape: {A.shape}")
    print(f"LL coefficients shape: {ll_coeffs.shape}")
    print(f"High-frequency mode: {'Sparse threshold training' if USE_SPARSE_HF else 'Zero-filled (LL only)'}")
    
    # Extract high-frequency coefficients based on mode
    if USE_SPARSE_HF:
        # Extract high-frequency coefficients with threshold-based sparsity
        # Only extract coefficients above a threshold to reduce model size
        # Use a threshold based on the standard deviation of all high-frequency coefficients
        
        # First, collect all high-frequency coefficients to compute statistics
        all_hf_coeffs = []
        for level_idx in range(1, len(coeffs)):
            cH, cV, cD = coeffs[level_idx]
            all_hf_coeffs.extend(cH.flatten())
            all_hf_coeffs.extend(cV.flatten())
            all_hf_coeffs.extend(cD.flatten())
        
        all_hf_coeffs = np.array(all_hf_coeffs)
        hf_std = np.std(all_hf_coeffs)
        hf_abs_mean = np.mean(np.abs(all_hf_coeffs))
        
        # Set threshold as a fraction of standard deviation
        threshold = THRESHOLD_FACTOR * hf_std
        
        print(f"High-frequency std: {hf_std:.2f}, abs mean: {hf_abs_mean:.2f}")
        print(f"Threshold for sparsity: {threshold:.2f}")
        
        high_freq_values = []
        high_freq_coords = []
        high_freq_shapes = []  # Store shapes for reconstruction
        
        global_idx = 0  # Global index across all high-frequency coefficients
        
        for level_idx in range(1, len(coeffs)):
            cH, cV, cD = coeffs[level_idx]
            high_freq_shapes.append((cH.shape, cV.shape, cD.shape))
            
            # Process each band (cH, cV, cD)
            for band_idx, band in enumerate([cH, cV, cD]):
                flat_band = band.flatten()
                
                # Find significant coefficients (above threshold)
                significant_mask = np.abs(flat_band) > threshold
                significant_values = flat_band[significant_mask]
                significant_indices = np.where(significant_mask)[0]
                
                # Store values and their global indices
                high_freq_values.extend(significant_values)
                
                # Create global indices that encode position in the flattened array
                for local_idx in significant_indices:
                    high_freq_coords.append(global_idx + local_idx)
                
                global_idx += len(flat_band)
        
        # Combine all significant high-frequency coefficients
        if high_freq_values:
            high_freq_combined = np.array(high_freq_values)
            high_freq_indices = np.array(high_freq_coords)
            total_hf_size = global_idx
            print(f"High-frequency coefficients total count: {total_hf_size}")
            print(f"Significant high-frequency coefficients: {len(high_freq_combined)} ({100*len(high_freq_combined)/total_hf_size:.2f}%)")
            print(f"Sparsity ratio: {100*(1 - len(high_freq_combined)/total_hf_size):.2f}%")
        else:
            high_freq_combined = np.array([])
            high_freq_indices = np.array([])
            total_hf_size = 0
    else:
        # No high-frequency training - will use zeros
        high_freq_combined = np.array([])
        high_freq_indices = np.array([])
        high_freq_shapes = []
        total_hf_size = 0
        
        # Still need to store shapes for reconstruction
        for level_idx in range(1, len(coeffs)):
            cH, cV, cD = coeffs[level_idx]
            high_freq_shapes.append((cH.shape, cV.shape, cD.shape))
        
        print("High-frequency coefficients will be filled with zeros (LL-only mode)")
    
    # Normalize LL coefficients for training stability
    ll_mean = ll_coeffs.mean()
    ll_std = ll_coeffs.std()
    ll_norm = (ll_coeffs - ll_mean) / (ll_std + 1e-8)
    
    # Normalize high-frequency coefficients separately
    if len(high_freq_combined) > 0:
        hf_mean = high_freq_combined.mean()
        hf_std = high_freq_combined.std()
        hf_norm = (high_freq_combined - hf_mean) / (hf_std + 1e-8)
    
    # Store the original coeffs and slices for reconstruction
    arr_full, slices = pywt.coeffs_to_array(coeffs)

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
    
    # Convert normalized LL coefficients to tensor
    coeffs_tensor = transforms.ToTensor()(ll_norm).float().to(device, dtype)

    # Initialize SIREN model for LL coefficients
    func_rep = Siren(
        dim_in=2,              # Input dimension (x,y coordinates)
        dim_hidden=LL_LAYER_SIZE, # Hidden layer size
        dim_out=1,            # Output dimension (coefficient value)
        num_layers=LL_NUM_LAYERS,
        final_activation=torch.nn.Identity(),
        w0_initial=30.0,      # Frequency scaling for first layer
        w0=30.0              # Frequency scaling for other layers
    )
    
    # Initialize SIREN model for high-frequency coefficients
    if len(high_freq_combined) > 0:
        func_rep_hf = Siren(
            dim_in=1,              # Input dimension (1D index)
            dim_hidden=HF_LAYER_SIZE, # Hidden layer size
            dim_out=1,            # Output dimension (coefficient value)
            num_layers=HF_NUM_LAYERS,
            final_activation=torch.nn.Identity(),
            w0_initial=30.0,
            w0=30.0
        )

    # Initialize trainer
    trainer = Trainer(func_rep, lr=2e-4)

    # Generate coordinates and features for training LL coefficients
    coordinates, features = util.to_coordinates_and_coeffs_features(coeffs_tensor)
    coordinates, features = coordinates.to(device, dtype), features.to(device, dtype)

    # Train the LL model
    print("\n=== Training LL (Low-frequency) Model ===")
    trainer.train(coordinates, features, num_iters=LL_ITERATIONS)
    print(f'Best LL training psnr: {trainer.best_vals["psnr"]:.2f}')
    
    # Train high-frequency model if we have high-frequency data
    if len(high_freq_combined) > 0:
        print("\n=== Training High-frequency Model (Sparse) ===")
        print(f"HF training samples: {len(high_freq_combined)}")
        print(f"HF normalized mean: {hf_norm.mean():.4f}, std: {hf_norm.std():.4f}")
        print(f"HF original mean: {high_freq_combined.mean():.4f}, std: {high_freq_combined.std():.4f}")
        
        trainer_hf = Trainer(func_rep_hf, lr=1e-4)  # Try lower learning rate
        
        # Create 1D coordinates using the actual indices where non-zero values exist
        hf_coords = torch.tensor(high_freq_indices.reshape(-1, 1), dtype=torch.float32).to(device, dtype)
        # Normalize indices to [-1, 1]
        hf_coords = (hf_coords / (total_hf_size - 1)) * 2 - 1
        
        hf_features = torch.tensor(hf_norm.reshape(-1, 1), dtype=torch.float32).to(device, dtype)
        
        print(f"HF coords range: [{hf_coords.min():.4f}, {hf_coords.max():.4f}]")
        print(f"HF features range: [{hf_features.min():.4f}, {hf_features.max():.4f}]")
        
        trainer_hf.train(hf_coords, hf_features, num_iters=HF_ITERATIONS)
        print(f'Best HF training psnr: {trainer_hf.best_vals["psnr"]:.2f}')

    # Calculate and log model size metrics
    model_size = util.model_size_in_bits(func_rep) / 8000.
    print(f'\nLL Model size: {model_size:.1f}kB')
    
    if len(high_freq_combined) > 0:
        model_size_hf = util.model_size_in_bits(func_rep_hf) / 8000.
        print(f'HF Model size: {model_size_hf:.1f}kB')
        print(f'Total Model size: {model_size + model_size_hf:.1f}kB')
    
    fp_bpp = util.bpp(model=func_rep, image=coeffs_tensor)
    print(f'Full precision bpp (LL): {fp_bpp:.2f}')

    # Log full precision results
    results['fp_bpp'].append(fp_bpp)
    results['fp_psnr'].append(trainer.best_vals['psnr'])

    # Save best models
    torch.save(trainer.best_model, os.path.join(LOG_DIR, 'best_model_dwt_ll.pt'))
    if len(high_freq_combined) > 0:
        torch.save(trainer_hf.best_model, os.path.join(LOG_DIR, 'best_model_dwt_hf.pt'))

    # Load best models and convert to half precision
    func_rep.load_state_dict(trainer.best_model)
    func_rep = func_rep.half().to('cuda')
    coordinates = coordinates.half().to('cuda')
    
    if len(high_freq_combined) > 0:
        func_rep_hf.load_state_dict(trainer_hf.best_model)
        func_rep_hf = func_rep_hf.half().to('cuda')
        hf_coords = hf_coords.half().to('cuda')

    # Calculate half precision metrics
    hp_bpp = util.bpp(model=func_rep, image=coeffs_tensor)
    results['hp_bpp'].append(hp_bpp)
    print(f'Half precision bpp: {hp_bpp:.2f}')

    with torch.no_grad():
        # Reconstruct LL coefficients from network
        coeffs_recon = func_rep(coordinates).reshape(coeffs_tensor.shape[1], coeffs_tensor.shape[2]).float()
        
        # Denormalize reconstructed LL coefficients
        coeffs_recon_denorm = coeffs_recon * (ll_std + 1e-8) + ll_mean
        
        # Calculate PSNR on LL coefficients
        original_ll_coeffs = torch.tensor(ll_coeffs).to(device, dtype)
        hp_psnr = util.get_clamped_psnr(original_ll_coeffs, coeffs_recon_denorm)
        
        # Reconstruct high-frequency coefficients if available
        if len(high_freq_combined) > 0:
            # Generate predictions for all positions (including zeros)
            # Create full index array for all high-frequency positions
            all_indices = torch.arange(total_hf_size, dtype=torch.float32).reshape(-1, 1).to(device)
            all_indices_norm = (all_indices / (total_hf_size - 1)) * 2 - 1
            all_indices_norm = all_indices_norm.half()
            
            # Predict all values (model will output near-zero for positions not trained)
            hf_recon_full = func_rep_hf(all_indices_norm).reshape(-1).float()
            # Denormalize
            hf_recon_denorm_full = hf_recon_full * (hf_std + 1e-8) + hf_mean
            hf_recon_np = hf_recon_denorm_full.cpu().numpy()
        
        # Save metrics
        results_file = os.path.join(LOG_DIR, 'metrics.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Convert reconstructed LL coefficients to numpy
        ll_recon_np = coeffs_recon_denorm.cpu().numpy()
        
        # Create a modified coefficient structure with reconstructed values
        coeffs_modified = [ll_recon_np]  # Start with reconstructed LL
        
        # Reconstruct high-frequency bands from the learned model
        if len(high_freq_combined) > 0:
            # Split the reconstructed high-frequency data back into cH, cV, cD
            idx = 0
            for i in range(len(high_freq_shapes)):
                cH_shape, cV_shape, cD_shape = high_freq_shapes[i]
                
                # Extract cH
                cH_size = np.prod(cH_shape)
                cH_recon = hf_recon_np[idx:idx+cH_size].reshape(cH_shape)
                idx += cH_size
                
                # Extract cV
                cV_size = np.prod(cV_shape)
                cV_recon = hf_recon_np[idx:idx+cV_size].reshape(cV_shape)
                idx += cV_size
                
                # Extract cD
                cD_size = np.prod(cD_shape)
                cD_recon = hf_recon_np[idx:idx+cD_size].reshape(cD_shape)
                idx += cD_size
                
                coeffs_modified.append((cH_recon, cV_recon, cD_recon))
        else:
            # Add zero coefficients for all high-frequency bands if no HF model
            for i in range(1, len(coeffs)):
                cH = np.zeros_like(coeffs[i][0])
                cV = np.zeros_like(coeffs[i][1])
                cD = np.zeros_like(coeffs[i][2])
                coeffs_modified.append((cH, cV, cD))
        
        coeffs_recon_list = coeffs_modified
        
        # Perform inverse DWT
        img_recon = pywt.waverec2(coeffs_recon_list, wavelet=WAVELET)
        
        # Save reconstructed image
        img_recon = np.clip(img_recon, 0, 255).astype(np.uint8)
        out_img = Image.fromarray(img_recon)
        out_img.save(OUTPUT_FILE)
        
        # Calculate and print image-space PSNR
        original_img = np.asarray(Image.open("kodak-dataset/kodim01.png").convert("L"))
        img_psnr = util.calc_psnr(original_img, img_recon)
        print(f'Image-space PSNR: {img_psnr:.2f}')
        
        print(f'Half precision PSNR: {hp_psnr:.2f}')
        results['hp_psnr'].append(hp_psnr)
        results['hp_calc_psnr'].append(float(img_psnr))

    print("\nDWT Sparse Results:", results)

if __name__ == "__main__":
    main()
