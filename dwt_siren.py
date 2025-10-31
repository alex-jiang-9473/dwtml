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
LEVELS = 1        # Number of DWT decomposition levels
WAVELET = "db4"   # Wavelet type: 'db1','db4','sym4', etc.
LOG_DIR = "results/dwt"
NUM_LAYERS = 20   # Number of layers in SIREN
LAYER_SIZE = 56   # Hidden layer size
ITERATIONS = 2000 # Training iterations
OUTPUT_FILE = f"dwt_siren_levels{LEVELS}_{WAVELET}_{NUM_LAYERS}_{LAYER_SIZE}_{ITERATIONS}.png"

# ---------------------------

def main():
    # 1) Load and prepare grayscale image
    img = Image.open("kodak-dataset/kodim01.png").convert("L")
    A = np.asarray(img, dtype=np.float32)

    # 2) Perform multi-level 2D DWT
    coeffs = pywt.wavedec2(A, wavelet=WAVELET, level=LEVELS)

    # 3) Convert coefficients to array and normalize
    arr, slices = pywt.coeffs_to_array(coeffs)

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
    trainer.train(coordinates, features, num_iters=ITERATIONS)
    print(f'Best training psnr: {trainer.best_vals["psnr"]:.2f}')

    # Log full precision results
    results['fp_bpp'].append(fp_bpp)
    results['fp_psnr'].append(trainer.best_vals['psnr'])

    # Save best model
    torch.save(trainer.best_model, os.path.join(LOG_DIR, 'best_model_dwt.pt'))

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
        
        # Save metrics
        results_file = os.path.join(LOG_DIR, 'metrics.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
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
        original_img = np.asarray(Image.open("kodak-dataset/kodim01.png").convert("L"))
        img_psnr = util.calc_psnr(original_img, img_recon)
        print(f'Image-space PSNR: {img_psnr:.2f}')
        
        print(f'Half precision PSNR: {hp_psnr:.2f}')
        results['hp_psnr'].append(hp_psnr)

    print("DWT Results:", results)

if __name__ == "__main__":
    main()