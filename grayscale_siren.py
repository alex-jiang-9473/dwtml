import os
import random
import json
import numpy as np
import torch
from PIL import Image
from training import Trainer
from siren import Siren
import util


# ---------------------------
# CONFIG
OUTPUT_FILE = "grayscale_reconstruction.png"
LOG_DIR = "results/grayscale"
NUM_LAYERS = 10
LAYER_SIZE = 28
ITERATIONS = 1000
# ---------------------------

# 1) Load an image (grayscale for simplicity)
img = Image.open("kodak-dataset/kodim01.png").convert("L")
A = np.asarray(img, dtype=np.float32)

# Normalize image to [0,1] range for better training stability
A = A / 255.0

# Set up torch and cuda
dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

seed = random.randint(1, int(1e6))
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Dictionary to register mean values (both full precision and half precision)
results = {'fp_bpp': [], 'hp_bpp': [], 'fp_psnr': [], 'hp_psnr': [], 'fp_calc_psnr': [], 'hp_calc_psnr': []}

# Create directory to store experiments
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Convert image to tensor
img_tensor = torch.tensor(A).unsqueeze(0).float().to(device, dtype)  # Add channel dimension

# Setup model
func_rep = Siren(
    dim_in=2,
    dim_hidden=LAYER_SIZE,
    dim_out=1,
    num_layers=NUM_LAYERS,
    final_activation=torch.nn.Sigmoid(),  # Use sigmoid to ensure output in [0,1]
    w0_initial=30.0,
    w0=30.0
)

trainer = Trainer(func_rep, lr=2e-4)

coordinates, features = util.to_coordinates_and_features(img_tensor)
coordinates, features = coordinates.to(device, dtype), features.to(device, dtype)

# Calculate model size. Divide by 8000 to go from bits to kB
model_size = util.model_size_in_bits(func_rep) / 8000.
print(f'Model size: {model_size:.1f}kB')
fp_bpp = util.bpp(model=func_rep, image=img_tensor)
print(f'Full precision bpp: {fp_bpp:.2f}')

# Train model
trainer.train(coordinates, features, num_iters=ITERATIONS)

print(f'Best training psnr: {trainer.best_vals["psnr"]:.2f}')

# Log full precision results
results['fp_bpp'].append(fp_bpp)
results['fp_psnr'].append(trainer.best_vals['psnr'])

# Save best model
torch.save(trainer.best_model, LOG_DIR + f'/best_model_grayscale.pt')

# Update current model to be best model
func_rep.load_state_dict(trainer.best_model)

func_rep = func_rep.half().to('cuda')
coordinates = coordinates.half().to('cuda')

# Calculate model size in half precision
hp_bpp = util.bpp(model=func_rep, image=img_tensor)
results['hp_bpp'].append(hp_bpp)
print(f'Half precision bpp: {hp_bpp:.2f}')

with torch.no_grad():
    # Reconstruct image
    img_recon = func_rep(coordinates).reshape(img_tensor.shape[1], img_tensor.shape[2]).float()
    img_recon_3d = img_recon.unsqueeze(0)
    
    # Calculate PSNR
    hp_psnr = util.get_clamped_psnr(img_tensor, img_recon_3d)
    
    # Save results and metrics
    results_file = LOG_DIR + '/metrics.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Convert to image and save
    vis_arr = (img_recon.cpu().numpy() * 255.0).astype(np.uint8)
    out_img = Image.fromarray(vis_arr)
    out_img.save(OUTPUT_FILE)
    
    # Calculate PSNR between original and reconstructed images
    original_img = np.asarray(Image.open("kodak-dataset/kodim01.png").convert("L"))
    img_psnr = util.calc_psnr(original_img, vis_arr)
    print(f'Image-space PSNR: {img_psnr:.2f}')
    
    print(f'Half precision psnr: {hp_psnr:.2f}')
    results['hp_psnr'].append(hp_psnr)
    results['hp_calc_psnr'].append(float(img_psnr))

print("Grayscale Results:", results)